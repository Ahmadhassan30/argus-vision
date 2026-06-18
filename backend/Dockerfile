# Argus Vision backend image (CPU-only PyTorch build).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System libraries required by OpenCV (cv2) at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
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

# Copy the rest of the backend source into the image.
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
