# Argus Vision backend image (CPU-only PyTorch build).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System libraries required by OpenCV (cv2) and LightGBM at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching.
COPY requirements.txt .
# Install the CPU-only PyTorch build explicitly from the PyTorch index first
# (far smaller than the default CUDA wheels), then the remaining deps from PyPI.
# The torch/torchvision pins match requirements.txt, so the second install sees
# them as already satisfied (2.2.0+cpu satisfies the ==2.2.0 pin).
RUN pip install torch==2.2.0 torchvision==0.17.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Phase 8: pin scikit-learn to the EXACT version used to fit the consensus scaler/model.
# joblib/pickle artifacts are sklearn-version-sensitive; a mismatch can silently change or
# break consensus predictions. Keep this explicit even though requirements.txt also pins it.
RUN pip install scikit-learn==1.6.1

# Copy the rest of the backend source into the image.
COPY . .

# Phase 9 hardening: run as a NON-ROOT user. Weight caches live under the user's home
# (HF_HOME/TORCH_HOME) because /root is not writable by a non-root user; /tmp/argus is the
# upload scratch dir. The docker-compose 'hf-cache' named volume mounts at /home/argus/.cache
# and inherits this directory's ownership on first creation, so the non-root user can write
# the downloaded EfficientNet/ViT weights.
ENV HF_HOME=/home/argus/.cache/huggingface \
    TORCH_HOME=/home/argus/.cache/torch
RUN useradd --create-home --uid 10001 argus \
    && mkdir -p /tmp/argus /home/argus/.cache/huggingface /home/argus/.cache/torch \
    && chown -R argus:argus /app /tmp/argus /home/argus
USER argus

EXPOSE 8000

# Production CMD has NO --reload. Dev hot-reload is added as a command override in
# docker-compose.yml (so the shipped image stays production-clean).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
