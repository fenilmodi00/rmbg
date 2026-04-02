import io
import numpy as np
from PIL import Image, ImageOps
from typing import Tuple, List, Optional
import httpx
import logging

logger = logging.getLogger(__name__)

def bytes_to_pil(data: bytes) -> Image.Image:
    """Converts image bytes to a PIL Image (JPEG/PNG only). Handles mobile EXIF rotation."""
    try:
        img = Image.open(io.BytesIO(data))
        # Handle mobile EXIF orientation
        img = ImageOps.exif_transpose(img)
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        logger.error(f"Error reading image bytes: {e}")
        raise ValueError("Cannot read image file. Ensure it is a valid JPEG or PNG.")

def pil_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    """Converts a PIL Image back into bytes."""
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()

def resize_for_inference(img: Image.Image, max_side: int = 1024) -> Image.Image:
    """Resizes an image maintaining aspect ratio so the largest side is max_side."""
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return img

def get_bounding_box(mask: np.ndarray, padding: int = 20) -> Tuple[int, int, int, int]:
    """
    Calculates a tight bounding box from a binary mask (0 or 255).
    Returns (x1, y1, x2, y2) with padding, clamped to mask dimensions.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # No object found in mask
        return 0, 0, mask.shape[1], mask.shape[0]
        
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    # Add padding and clamp
    h, w = mask.shape
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    return int(x1), int(y1), int(x2), int(y2)

def mask_to_alpha(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Applies a binary numpy mask as the alpha channel to an RGB image."""
    # Ensure mask is 0-255 uint8
    if mask.max() <= 1.0:
         mask = (mask * 255).astype(np.uint8)
    else:
         mask = mask.astype(np.uint8)
         
    alpha_mask = Image.fromarray(mask, mode="L")
    
    # Create RGBA copy
    rgba = image.copy().convert("RGBA")
    rgba.putalpha(alpha_mask)
    return rgba

def add_white_background(rgba_image: Image.Image) -> Image.Image:
    """Composites an RGBA image onto a solid white background."""
    white_bg = Image.new("RGB", rgba_image.size, (255, 255, 255))
    white_bg.paste(rgba_image, (0, 0), rgba_image)
    return white_bg

async def download_image(url: str) -> bytes:
    """Downloads an image from a URL."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {e}")
            raise ValueError(f"Could not download image: {str(e)}")
