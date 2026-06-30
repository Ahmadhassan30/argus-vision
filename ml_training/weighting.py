"""Class-imbalance weighting for Argus Vision — the single source of truth.

Defines :func:`effective_number_weights` (Cui et al., 2019, "Class-Balanced Loss
Based on Effective Number of Samples"). The SAME returned weight array is used for
BOTH the ``WeightedRandomSampler`` per-sample weights and the ``FocalLoss`` ``alpha``,
in notebooks 01/02 and in :mod:`dataset`, so the sampler and the loss can never drift
apart. Replaces the previous 1/sqrt(count) weighting, which under-corrected the heavy
ISIC-8 imbalance (a 51:1 raw ratio was only softened to roughly 7:1 of sampling mass).

Torch-free on purpose: it returns a NumPy array when NumPy is available (so the
notebooks' fancy-indexing ``weights[labels]`` works on Kaggle) and a plain ``list``
otherwise (so the weighting can be unit-tested in a minimal environment). The two paths
are mathematically identical.
"""

from __future__ import annotations

from typing import Sequence

try:
    # config.py imports torch at module load; in a minimal env (no torch) fall back to
    # the same documented default so this module stays importable for unit testing.
    from config import EFFECTIVE_NUMBER_BETA
except Exception:  # pragma: no cover - exercised only without torch installed
    EFFECTIVE_NUMBER_BETA = 0.99


def effective_number_weights(class_counts: Sequence[float], beta: float = EFFECTIVE_NUMBER_BETA):
    """Effective-number-of-samples class weights, normalized to mean 1.0.

    ``effective_num_c = 1 - beta**count_c``; ``w_c = (1 - beta) / effective_num_c``;
    then rescaled so the weights sum to ``len(class_counts)`` (mean 1.0), which keeps
    the overall loss/sampling scale comparable to unweighted training.

    Args:
        class_counts: Per-class sample counts, in canonical class order.
        beta: Effective-number hyper-parameter in ``[0, 1)`` (default
            :data:`config.EFFECTIVE_NUMBER_BETA`). Higher ``beta`` => stronger
            up-weighting of rare classes.

    Returns:
        A NumPy ``float64`` array (if NumPy is importable) or a ``list[float]`` of
        per-class weights, normalized so their mean is 1.0.
    """
    try:
        import numpy as np

        counts = np.asarray(class_counts, dtype=np.float64)
        # Guard empty classes so 1/effective_num stays finite (count 0 -> weight 0).
        effective_num = 1.0 - np.power(beta, counts)
        weights = np.where(effective_num > 0, (1.0 - beta) / np.where(effective_num > 0, effective_num, 1.0), 0.0)
        total = weights.sum()
        if total > 0:
            weights = weights / total * len(counts)
        return weights
    except ImportError:
        counts = [float(c) for c in class_counts]
        eff = [1.0 - beta ** c for c in counts]
        weights = [((1.0 - beta) / e if e > 0 else 0.0) for e in eff]
        total = sum(weights)
        n = len(counts)
        if total > 0:
            weights = [w / total * n for w in weights]
        return weights
