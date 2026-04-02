import time
import logging
import base64
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io

from config import settings
import pipeline
import schemas
import utils

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress noise from dependencies for production (The "Lightning" clean output)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.WARNING)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event for preloading models on startup."""
    logger.info("Starting Outfit Extractor (Production Grade)...")
    try:
        pipeline.load_models()
        logger.info("⚡ Models ready for Lightning Inference.")
    except Exception as e:
        logger.error(f"Failed to preload models: {e}")
    yield
    logger.info("Shutting down service...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to log request duration."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} handled in {process_time:.4f}s")
    response.headers["X-Process-Time"] = str(process_time)
    return response

def check_models_ready():
    """Returns 503 if models are still loading."""
    if not pipeline.is_ready():
        return Response(
            content="Models are still loading. Please try again in a few seconds.",
            status_code=503,
            headers={"Retry-After": "5"}
        )
    return None

@app.get("/healthz", status_code=200)
async def liveness():
    """Liveness probe: confirms the server is up."""
    return {"status": "ok"}

@app.get("/readyz")
async def readiness():
    """Readiness probe: confirms models are loaded and ready for inference."""
    if not pipeline.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Models are still loading",
            headers={"Retry-After": "5"}
        )
    return {"status": "ready"}

@app.get("/health", response_model=schemas.HealthResponse)
async def health():
    """Detailed health check for manual oversight."""
    return {
        "status": "online" if pipeline.is_ready() else "loading",
        "segformer_loaded": pipeline.is_ready(),
        "birefnet_loaded": pipeline.is_ready(),
        "device": settings.DEVICE,
        "version": settings.VERSION
    }

@app.post("/extract-outfit", response_model=schemas.ExtractionResponse)
async def extract_outfit_api(image: UploadFile = File(...)):
    # Cold start guard
    wait_resp = check_models_ready()
    if wait_resp: return wait_resp

    # Validation (Added AVIF support)
    if image.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/avif"]:
        raise HTTPException(status_code=400, detail=f"Only JPEG, PNG, or AVIF accepted. Got {image.content_type}")
    
    content = await image.read()
    if len(content) > settings.MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPError(status_code=400, detail=f"File too large (max {settings.MAX_IMAGE_SIZE_MB}MB)")

    try:
        result = pipeline.extract_outfit(content)
        
        # Convert to Base64 for JSON response
        t_b64 = base64.b64encode(result["transparent_png"]).decode("utf-8")
        w_b64 = base64.b64encode(result["white_bg_png"]).decode("utf-8")
        
        return {
            "transparent_image_b64": t_b64,
            "white_bg_image_b64": w_b64,
            "labels_found": result["labels_found"],
            "inference_time_ms": result["inference_time_ms"],
            "image_width": result["width"],
            "image_height": result["height"]
        }
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.post("/extract-outfit/url", response_model=schemas.ExtractionResponse)
async def extract_outfit_url(request: schemas.ExtractFromURLRequest):
    # Cold start guard
    wait_resp = check_models_ready()
    if wait_resp: return wait_resp

    try:
        image_bytes = await utils.download_image(str(request.image_url))
        result = pipeline.extract_outfit(image_bytes)
        
        t_b64 = base64.b64encode(result["transparent_png"]).decode("utf-8")
        w_b64 = base64.b64encode(result["white_bg_png"]).decode("utf-8")
        
        return {
            "transparent_image_b64": t_b64,
            "white_bg_image_b64": w_b64,
            "labels_found": result["labels_found"],
            "inference_time_ms": result["inference_time_ms"],
            "image_width": result["width"],
            "image_height": result["height"]
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.post("/extract-outfit/raw")
async def extract_outfit_raw(output: str = "transparent", image: UploadFile = File(...)):
    # Cold start guard
    wait_resp = check_models_ready()
    if wait_resp: return wait_resp

    content = await image.read()
    try:
        result = pipeline.extract_outfit(content)
        key = "transparent_png" if output == "transparent" else "white_bg_png"
        return Response(content=result[key], media_type="image/png")
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Internal processing error")

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Outfit Extractor</title>
        <style>
            body { font-family: -apple-system, system-ui, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 2rem; }
            .container { max-width: 900px; margin: 0 auto; }
            h1 { color: #818cf8; font-weight: 800; border-bottom: 1px solid #1e293b; padding-bottom: 1rem; }
            .card { background: #1e293b; padding: 2rem; border-radius: 1rem; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.4); }
            input[type=file] { background: #0f172a; border: 1px dashed #475569; padding: 1rem; width: 100%; border-radius: 0.5rem; margin-bottom: 1rem; color: #94a3b8; }
            button { background: #6366f1; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 0.5rem; font-weight: 600; cursor: pointer; display: block; width: 100%; }
            button:hover { background: #4f46e5; }
            button:disabled { background: #334155; cursor: not-allowed; }
            #results { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 2rem; display: none; }
            .result-item { text-align: center; }
            .result-item img { max-width: 100%; border-radius: 0.5rem; background: #0f172a; padding: 0.5rem; border: 1px solid #334155; }
            .status { margin-top: 1rem; font-size: 0.875rem; color: #94a3b8; text-align: center; }
            #loading-overlay { display: none; margin-top: 1rem; text-align: center; color: #818cf8; font-weight: 500; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Outfit Extractor</h1>
            <div class="card">
                <input type="file" id="imageInput" accept="image/jpeg,image/png">
                <button id="extractBtn">Extract Outfit</button>
                <div id="loading-overlay">Removing background & extracting clothing...</div>
                <div class="status" id="statusText">Upload a fashion photo to begin.</div>
            </div>
            
            <div id="results">
                <div class="result-item">
                    <h3>Transparent PNG</h3>
                    <img id="transImg" src="" />
                </div>
                <div class="result-item">
                    <h3>White Background</h3>
                    <img id="whiteImg" src="" />
                </div>
            </div>
        </div>

        <script>
            const btn = document.getElementById('extractBtn');
            const input = document.getElementById('imageInput');
            const results = document.getElementById('results');
            const status = document.getElementById('statusText');
            const loading = document.getElementById('loading-overlay');

            btn.onclick = async () => {
                if (!input.files[0]) return alert("Select an image first");

                btn.disabled = true;
                results.style.display = 'none';
                loading.style.display = 'block';
                status.innerText = "Processing Pipeline Step 1 (Segmentation) + Step 2 (Edge Refinement)...";

                const formData = new FormData();
                formData.append('image', input.files[0]);

                try {
                    const response = await fetch('/extract-outfit', {
                        method: 'POST',
                        body: formData
                    });

                    if (response.status === 503) {
                        status.innerText = "Error: Models are cold-starting. Retrying in 5 seconds...";
                        setTimeout(btn.onclick, 5000);
                        return;
                    }

                    if (!response.ok) {
                        const err = await response.json();
                        throw new Error(err.detail || "Server error");
                    }

                    const data = await response.json();
                    document.getElementById('transImg').src = `data:image/png;base64,${data.transparent_image_b64}`;
                    document.getElementById('whiteImg').src = `data:image/png;base64,${data.white_bg_image_b64}`;
                    
                    results.style.display = 'grid';
                    status.innerText = `Found: ${data.labels_found.join(', ')} | Time: ${data.inference_time_ms}ms`;
                } catch (e) {
                    status.innerText = "Error: " + e.message;
                } finally {
                    loading.style.display = 'none';
                    btn.disabled = false;
                }
            };
        </script>
    </body>
    </html>
    """
