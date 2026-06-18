"""Central training configuration for Argus Vision.

This module is the single source of truth for hyper-parameters, dataset
constants and device selection used across the ML training pipeline (dataset
construction, loss functions, model heads and the training loops). All values
mirror the Argus Vision shared contract: the ISIC-8 class ordering, the input
image size and the ImageNet normalization statistics are intentionally kept
identical to the values used by the backend inference stack so that the
exported checkpoints behave consistently at serving time.
"""

from __future__ import annotations

import torch

# --- ISIC-8 class taxonomy (exact contract order, index 0..7) --------------
ISIC_CLASSES: list[str] = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
NUM_CLASSES: int = 8

# --- Image / preprocessing constants ---------------------------------------
IMAGE_SIZE: int = 224
IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: list[float] = [0.229, 0.224, 0.225]

# --- Data loading -----------------------------------------------------------
BATCH_SIZE: int = 32
NUM_WORKERS: int = 4

# --- Device selection -------------------------------------------------------
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

# --- Loss hyper-parameters --------------------------------------------------
FOCAL_LOSS_GAMMA: float = 2.0
LABEL_SMOOTHING: float = 0.1

# --- Training schedule ------------------------------------------------------
MAX_EPOCHS_HEAD: int = 5
MAX_EPOCHS_FINETUNE: int = 15

# --- Learning rates ---------------------------------------------------------
LR_HEAD: float = 1e-3
LR_BACKBONE: float = 1e-5
LR_ATTN: float = 1e-6

# --- Checkpoints ------------------------------------------------------------
CHECKPOINT_DIR: str = "./checkpoints"
