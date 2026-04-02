# --- STAGE 1: Builder ---
FROM python:3.11-slim AS builder

# Set environment variables for build
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first for layer caching
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download and cache models in a specific directory
# We set HF_HOME to ensure the runner finds them in the exact same spot.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "import torch; \
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, AutoModelForImageSegmentation; \
print('Pre-downloading SegFormer...'); \
SegformerImageProcessor.from_pretrained('sayeed99/segformer_b3_clothes'); \
SegformerForSemanticSegmentation.from_pretrained('sayeed99/segformer_b3_clothes'); \
print('Pre-downloading BiRefNet-lite...'); \
AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet_lite', trust_remote_code=True);"

# --- STAGE 2: Runner ---
FROM python:3.11-slim AS runner

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV HF_HOME=/app/.cache/huggingface
ENV DEVICE=cpu

WORKDIR /app

# Install runtime system dependencies (OpenCV requirements)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-privileged user for security
RUN groupadd -r appgroup && useradd -r -g appgroup -s /sbin/nologin appuser

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy pre-downloaded models from builder
COPY --from=builder --chown=appuser:appgroup /app/.cache /app/.cache

# Copy source code
COPY --chown=appuser:appgroup . .

# Change to non-root user
USER appuser

EXPOSE 8080

# Cloud Run uses SIGTERM for graceful shutdown, which Uvicorn handles by default.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
