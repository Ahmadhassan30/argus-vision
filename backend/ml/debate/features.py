"""Canonical 23-dimensional consensus feature extractor (serving path).

This replaces the removed Groq LLM debate-text + 788-dim sentence-embedding
pipeline. The consensus head now consumes a 23-dim pure numerical vector built
from the two agents' softmax distributions, their disagreement statistics, and
their spatial-attention agreement.

The implementation here is duplicated **verbatim** from
``ml_training/debate_text_utils.py`` and from the Kaggle notebooks
(04_train_consensus.ipynb / 05_evaluation.ipynb). All three must stay byte-for-byte
identical or training/eval/serving features silently disagree. See that module's
docstring for the full feature layout.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy as scipy_entropy
from scipy.spatial.distance import jensenshannon

# ISIC-8 class names in canonical (index 0..7) order.
CLASS_NAMES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# Dimensionality of the consensus feature vector. Never change without retraining
# the consensus MLP and re-fitting the StandardScaler.
FEATURE_DIM = 23

# Human-readable names of the 23 features, in order (for diagnostics/logging).
FEATURE_NAMES = (
    [f"pA_{c}" for c in CLASS_NAMES]
    + [f"pB_{c}" for c in CLASS_NAMES]
    + ["js_div", "entropy_a", "entropy_b", "max_prob_delta",
       "attn_iou", "attn_entropy_a", "attn_entropy_b"]
)


def extract_consensus_features(
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    attn_map_a: np.ndarray = None,
    attn_map_b: np.ndarray = None,
) -> np.ndarray:
    """Build the 23-d consensus feature vector for one image.

    Args:
        prob_a: Agent A softmax probabilities, shape ``(8,)`` (ISIC-8 order).
        prob_b: Agent B softmax probabilities, shape ``(8,)`` (ISIC-8 order).
        attn_map_a: Agent A 2D attention/saliency map, or ``None`` if unavailable.
        attn_map_b: Agent B 2D attention/saliency map, or ``None`` if unavailable.

    Returns:
        A ``float32`` ``np.ndarray`` of shape ``(23,)``. The three attention
        features are ``0.0`` when either attention map is ``None`` (the
        non-debate fast path), which is a legitimate value, not a missing one.
    """
    prob_a = np.asarray(prob_a, dtype=np.float64).flatten()
    prob_b = np.asarray(prob_b, dtype=np.float64).flatten()

    prob_a = np.clip(prob_a, 1e-9, 1.0)
    prob_a /= prob_a.sum()
    prob_b = np.clip(prob_b, 1e-9, 1.0)
    prob_b /= prob_b.sum()

    js_div = float(jensenshannon(prob_a, prob_b) ** 2)
    entropy_a = float(scipy_entropy(prob_a, base=2))
    entropy_b = float(scipy_entropy(prob_b, base=2))
    max_prob_delta = float(np.max(np.abs(prob_a - prob_b)))

    if attn_map_a is not None and attn_map_b is not None:
        a = np.asarray(attn_map_a, dtype=np.float32)
        b = np.asarray(attn_map_b, dtype=np.float32)
        a = (a - a.min()) / (a.max() - a.min() + 1e-9)
        b = (b - b.min()) / (b.max() - b.min() + 1e-9)
        mask_a = (a >= 0.5).astype(np.float32)
        mask_b = (b >= 0.5).astype(np.float32)
        intersection = (mask_a * mask_b).sum()
        union = np.clip(mask_a + mask_b, 0, 1).sum()
        attn_iou = float(intersection / (union + 1e-9))
        a_flat = a.flatten() + 1e-9
        a_flat /= a_flat.sum()
        b_flat = b.flatten() + 1e-9
        b_flat /= b_flat.sum()
        attn_entropy_a = float(scipy_entropy(a_flat, base=2))
        attn_entropy_b = float(scipy_entropy(b_flat, base=2))
    else:
        attn_iou = 0.0
        attn_entropy_a = 0.0
        attn_entropy_b = 0.0

    feat = np.concatenate([
        prob_a,
        prob_b,
        [js_div, entropy_a, entropy_b, max_prob_delta,
         attn_iou, attn_entropy_a, attn_entropy_b],
    ]).astype(np.float32)

    assert feat.shape == (FEATURE_DIM,), \
        f"Feature dim mismatch: got {feat.shape}, expected ({FEATURE_DIM},)"
    assert not np.any(np.isnan(feat)), "NaN in feature vector"
    assert not np.any(np.isinf(feat)), "Inf in feature vector"
    return feat
