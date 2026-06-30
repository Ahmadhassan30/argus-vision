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

# --- Class-imbalance weighting ----------------------------------------------
# Effective-number-of-samples weighting (Cui et al., 2019): w_c = (1 - beta) /
# (1 - beta**count_c), normalized to mean 1. The SAME array drives BOTH the
# WeightedRandomSampler per-sample weights AND the FocalLoss alpha (so they cannot
# drift apart), replacing the old 1/sqrt(count) weighting.
#
# CHOICE OF BETA — measured on a 100k-draw sampler simulation over the real ISIC-2019
# counts (max:min = ratio of most- to least-sampled class; ideal 1.0):
#   beta=0.99   -> 50:1  NO-OP. 1/(1-beta)=100 is the effective-sample saturation, but
#                        EVERY ISIC class has >100 samples, so all get ~equal weight. This
#                        is WORSE than the old 1/sqrt (7.4:1). Do not use at this scale.
#   beta=0.999  -> 11:1  CHOSEN. A real, substantial rebalancing that is not a no-op,
#                        without pushing into small-pool over-repetition.
#   beta=0.9999 -> 1.8:1 REJECTED. Near-uniform means the 239 physical DF images get drawn
#                        ~as often as NV's 12,875 — heavy repetition of a tiny pool, not new
#                        information. Because this same array also scales the focal loss,
#                        rare classes would be over-sampled AND over-weighted simultaneously
#                        — an overfitting + calibration risk we can't afford (ECE ~10.9%).
# Effective-number only rebalances once 1/(1-beta) exceeds the MAJORITY count (~12,875).
EFFECTIVE_NUMBER_BETA: float = 0.999

# --- Training schedule ------------------------------------------------------
MAX_EPOCHS_HEAD: int = 5
# Phase-2 fine-tuning uses early stopping up to PHASE2_MAX_EPOCHS with PHASE2_PATIENCE.
# These are the single source of truth so the notebooks can't drift from config
# again (notebook 01 previously hardcoded 40 epochs / patience 8 while this file
# still said 15 — the notebooks now read these constants instead).
PHASE2_MAX_EPOCHS: int = 40
PHASE2_PATIENCE: int = 8
# Backwards-compatible alias (kept for any importer using the old name).
MAX_EPOCHS_FINETUNE: int = PHASE2_MAX_EPOCHS

# --- Decoupled training (Phase 4, Kang et al. 2020) -------------------------
# "joint"     = single-stage class-balanced fine-tune (the Phase-3 path).
# "decoupled" = Stage A (instance-balanced sampling, UNWEIGHTED loss, full network —
#               learn good features) then Stage B (freeze backbone, class-balanced
#               sampler + weighted focal loss, retrain ONLY the head — fix the boundary).
# A/B the two on Kaggle; do not assume which wins.
TRAINING_MODE: str = "decoupled"
STAGE_A_EPOCHS: int = 30  # representation learning (early-stops via PHASE2_PATIENCE)
STAGE_B_EPOCHS: int = 10  # classifier re-balancing on the frozen backbone

# Logit adjustment (Menon et al. 2021) — applied to logits before the loss using the
# EMPIRICAL (un-rebalanced) class priors. Toggleable INDEPENDENTLY of TRAINING_MODE so
# the two ideas can be tested in isolation or combined.
USE_LOGIT_ADJUSTMENT: bool = False
LOGIT_ADJUSTMENT_TAU: float = 1.0

# --- Learning rates ---------------------------------------------------------
LR_HEAD: float = 1e-3
LR_BACKBONE: float = 1e-5
LR_ATTN: float = 1e-6

# --- Checkpoints ------------------------------------------------------------
CHECKPOINT_DIR: str = "./checkpoints"
