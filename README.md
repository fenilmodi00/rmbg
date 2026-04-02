# Outfit Extractor API (CPU-Only)

A production-ready FastAPI application that extracts clothing from fashion photos using a two-step, high-precision pipeline. Optimized for CPU-only serverless deployment on **Google Cloud Run**.

---

## 🚀 Two-Step Pipeline Architecture

1.  **Step 1: SegFormer-B3** (`sayeed99/segformer_b3_clothes`)
    *   Performs initial semantic segmentation into 18 clothing labels.
    *   Filters for specific clothing items (Hats, Upper-clothes, Skirts, Pants, Dresses, etc.).
    *   Generates a tight bounding box around the detected outfit.
2.  **Step 2: BiRefNet-lite** (`ZhengPeng7/BiRefNet_lite`)
    *   Runs high-resolution (1024x1024) edge refinement on the cropped bounding box.
    *   Produces a sharp, professional-grade transparent mask.
3.  **Result**: Clean, high-quality transparent PNG and white-background variants.

---

## 🛠️ Local Development

### 1. Prerequisites
- Python 3.11+
- [Git LFS](https://git-lfs.github.com/) (recommended for model downloading)

### 2. Setup
```bash
# Clone the repository
# git clone <your-repo-url>
# cd outfit-extractor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install CPU-specific PyTorch (Recommended for speed/size)
pip install torch torchvision --index-url https://download.mpg.de/whl/cpu

# Install other dependencies
pip install -r requirements.txt
```

### 3. Run Locally
```bash
# Run with uvicorn
uvicorn main:app --port 8000 --reload
```

---

## 🐳 Docker (Local Test)

```bash
# Build the image (includes model pre-caching)
docker build -t outfit-extractor .

# Run the container
# Note: Maps internal 8080 to external 8000
docker run -p 8000:8080 outfit-extractor
```

---

## ☁️ Deploy to Google Cloud Run

This project includes a `cloudbuild.yaml` for zero-touch CI/CD or manual deployment.

### Manual Deployment via CLI
```bash
gcloud run deploy outfit-extractor \
  --source . \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 120 \
  --concurrency 1 \
  --min-instances 0 \
  --max-instances 10 \
  --allow-unauthenticated
```

---

## 🧪 Testing

```bash
# Run unit and integration tests
pytest tests/
```

---

## 📖 API Usage

### Interactive UI
Open `http://localhost:8000` in your browser for a built-in testing interface.

### POST /extract-outfit
Extract from a multipart file upload. Returns JSON with base64 images.
```bash
curl -X POST http://localhost:8000/extract-outfit \
  -F "image=@fashion_photo.jpg" | python -m json.tool
```

### POST /extract-outfit/url
Extract from a public image URL.
```bash
curl -X POST http://localhost:8000/extract-outfit/url \
  -H "Content-Type: application/json" \
  -d '{"image_url": "https://example.com/photo.jpg"}'
```

### POST /extract-outfit/raw
Stream the raw PNG result directly (useful for debugging).
```bash
curl -X POST "http://localhost:8000/extract-outfit/raw?output=transparent" \
  -F "image=@fashion_photo.jpg" --output result.png
```

---

## 📈 Performance & Cold Starts
- **Initialization**: 15–20 seconds (Models cached in Docker image, must be loaded into RAM). 
- **Cold Start Handler**: Returns HTTP 503 with `Retry-After: 5` header while loading.
- **Inference Time**: 2–5 seconds per image on 2 vCPUs.
- **Memory Usage**: ~1.2 GB RAM (Safe under 2 GiB limit).

---

## 📜 License
MIT
