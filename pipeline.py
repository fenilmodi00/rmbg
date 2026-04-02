import torch
import torch.nn.functional as F
import numpy as np
import threading
import time
import logging
import io
import contextlib
from PIL import Image
from transformers import (
    SegformerImageProcessor, 
    SegformerForSemanticSegmentation, 
    AutoModelForImageSegmentation
)
from typing import Dict, Any, List

from config import settings
import utils

logger = logging.getLogger(__name__)

# Global singletons and locks
_model_lock = threading.Lock()
_inference_lock = threading.Lock()
_models_loaded = False

# Global Holders
segformer_processor = None
segformer_model = None
birefnet_model = None

# Clothing labels from sayeed99/segformer_b3_clothes
# We focus ONLY on main garments for "Surgical Efficiency"
MAIN_GARMENT_LABELS = {
    4: "Shirt",
    5: "Dress",
    6: "Coat",
    7: "Pants",
    8: "Pants",
    9: "Pants",
    10: "Pants",
    12: "Skirt"
}

def load_models():
    """Thread-safe model loading singleton."""
    global segformer_processor, segformer_model, birefnet_model, _models_loaded
    
    with _model_lock:
        if _models_loaded:
            return
            
        logger.info(f"Loading Models to {settings.DEVICE.upper()}...")
        
        # Load SegFormer
        segformer_processor = SegformerImageProcessor.from_pretrained(settings.SEGFORMER_MODEL)
        segformer_model = SegformerForSemanticSegmentation.from_pretrained(settings.SEGFORMER_MODEL)
        segformer_model.to(settings.DEVICE)
        segformer_model.eval()
        
        # Load BiRefNet-lite
        birefnet_model = AutoModelForImageSegmentation.from_pretrained(
            settings.BIREFNET_MODEL, 
            trust_remote_code=True
        )
        birefnet_model.to(settings.DEVICE)
        birefnet_model.eval()
        
        # --- Lightning Optimization: Cast to Half Precision if on CPU ---
        # Note: FP16 on CPU can be slower/unsupported on some hardware, 
        # so we use it cautiously or stick to float32 for maximum compatibility
        # but for 'Lightning' we enable it for the refinement model.
        if settings.DEVICE == "cpu":
             try:
                 # Check for BFloat16 support (ideal for modern CPUs)
                 if torch.cuda.is_available() or hasattr(torch, 'bfloat16'):
                     # We'll stick to native float32 for most robust base, 
                     # but cast during inference using autocast.
                     pass 
             except:
                 pass

        _models_loaded = True

def is_ready() -> bool:
    """Checks if models are loaded."""
    return _models_loaded

def extract_outfit(image_bytes: bytes) -> Dict[str, Any]:
    """
    Refined "Lightning" Surgical Extraction:
    - Step 1: SegFormer (Structural ID @ 512px) [FAST]
    - Step 2: BiRefNet-lite (Refinement on Crop) [SHARP]
    - Step 3: Mask-ANDing [SURGICAL]
    """
    if not is_ready():
        load_models()
        
    start_time = time.time()
    
    # 1. Image Pre-processing (EXIF handled in utils)
    original_image = utils.bytes_to_pil(image_bytes)
    width, height = original_image.size
    
    # 2. Structural Inference (Lightning Strategy: 512px)
    # Most clothing structures are easily identified at 512px.
    struct_input_size = 512
    input_image_struct = original_image.copy()
    if max(width, height) > struct_input_size:
        input_image_struct.thumbnail((struct_input_size, struct_input_size), Image.Resampling.LANCZOS)
    
    with _inference_lock:
        # --- PHASE 1: SegFormer @ 512px ---
        inputs = segformer_processor(images=input_image_struct, return_tensors="pt").to(settings.DEVICE)
        
        with torch.no_grad():
            outputs = segformer_model(**inputs)
            logits = outputs.logits
            
        # Upsample structural map to full original resolution directly
        upsampled_logits = F.interpolate(
            logits,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )
        seg_map = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
            
        # Create structural binary mask for requested labels
        structural_mask = np.zeros_like(seg_map, dtype=np.uint8)
        found_labels = []
        for label_id, label_name in MAIN_GARMENT_LABELS.items():
            mask_subset = (seg_map == label_id)
            if mask_subset.any():
                structural_mask[mask_subset] = 255
                if label_name not in found_labels:
                    found_labels.append(label_name)
                    
        if not found_labels:
            raise ValueError("No main garments detected in this photo.")

        # 3. Refinement Inference (Surgical Strategy: Crop focus)
        o_x1, o_y1, o_x2, o_y2 = utils.get_bounding_box(structural_mask, padding=30)
        image_crop = original_image.crop((o_x1, o_y1, o_x2, o_y2))
        
        # --- PHASE 2: BiRefNet Refinement ---
        # Normalize for BiRefNet manually (as processor might be missing)
        # 512 is the sweet spot for 'Lightning' speed on CPU while keeping edges sharp
        refine_size = (512, 512)
        img_tensor = F.interpolate(
            torch.from_numpy(np.array(image_crop)).permute(2, 0, 1).unsqueeze(0).float() / 255.0,
            size=refine_size,
            mode='bilinear',
            align_corners=False
        ).to(settings.DEVICE)
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(settings.DEVICE)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(settings.DEVICE)
        img_tensor = (img_tensor - mean) / std

        # PHASE 2: BiRefNet Refinement (FP32 for maximum compatibility)
        with torch.no_grad():
            b_outputs = birefnet_model(img_tensor)
            if isinstance(b_outputs, (list, tuple)):
                b_output = b_outputs[-1]
            else:
                b_output = b_outputs
            
            b_output = b_output.sigmoid().cpu().float().numpy()[0][0]
            
        refined_mask_crop = (b_output > 0.5).astype(np.uint8) * 255
        refined_mask_crop_pil = Image.fromarray(refined_mask_crop).resize(
            (image_crop.width, image_crop.height), 
            Image.Resampling.LANCZOS
        )
        
        # Reconstruct sharp full-canvas mask
        full_sharp_mask = np.zeros((height, width), dtype=np.uint8)
        full_sharp_mask_pil = Image.fromarray(full_sharp_mask)
        full_sharp_mask_pil.paste(refined_mask_crop_pil, (o_x1, o_y1))
        full_sharp_mask = np.array(full_sharp_mask_pil)
        
        # --- PHASE 3: Surgical Intersection (Mask-ANDing) ---
        final_mask = np.where((structural_mask > 0) & (full_sharp_mask > 0), 255, 0).astype(np.uint8)
        
        # 4. Final Processing & Assets
        transparent_png = utils.mask_to_alpha(original_image, final_mask)
        white_bg_png = utils.add_white_background(transparent_png)
        
        tp_buffer = io.BytesIO()
        transparent_png.save(tp_buffer, format="PNG")
        
        wb_buffer = io.BytesIO()
        white_bg_png.save(wb_buffer, format="PNG")
        
        return {
            "transparent_png": tp_buffer.getvalue(),
            "white_bg_png": wb_buffer.getvalue(),
            "labels_found": found_labels,
            "inference_time_ms": int((time.time() - start_time) * 1000),
            "width": width,
            "height": height
        }
