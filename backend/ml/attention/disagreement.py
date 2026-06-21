"""Cross-agent attention disagreement and contested-region extraction.

Given two 224x224 saliency maps (Agent A's Grad-CAM++ and Agent B's attention
rollout), this module quantifies *where* the agents look differently and
extracts a bounding box around the most contested region. The resulting map and
region statistics feed the WebSocket ``attention_computed`` event (the UI
overlays). The 23-dim consensus feature vector derives its three attention
features directly from the two raw saliency maps via
``ml.debate.features.extract_consensus_features`` rather than from these region
statistics.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from core.models import BoundingBox

IMAGE_SIZE: int = 224


def _min_max_normalize(array: np.ndarray) -> np.ndarray:
    """Min-max normalize a 2-D array into ``[0, 1]``.

    Args:
        array: Any real-valued array.

    Returns:
        A ``float32`` copy scaled to ``[0, 1]``. A constant input yields all
        zeros (no spurious activation).
    """
    arr = np.asarray(array, dtype=np.float32)
    minimum = float(arr.min())
    maximum = float(arr.max())
    span = maximum - minimum
    if span < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - minimum) / span).astype(np.float32)


def _region_stats(normalized_map: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Compute mean / std / max statistics of a map within a boolean mask.

    Args:
        normalized_map: A ``[0, 1]``-normalized saliency map.
        mask: A boolean mask selecting the region of interest.

    Returns:
        A dict with ``"mean"``, ``"std"`` and ``"max"`` as Python floats. If the
        mask is empty all statistics are ``0.0``.
    """
    if not bool(mask.any()):
        return {"mean": 0.0, "std": 0.0, "max": 0.0}
    selected = normalized_map[mask]
    return {
        "mean": float(selected.mean()),
        "std": float(selected.std()),
        "max": float(selected.max()),
    }


def compute_disagreement(
    heatmap_a: np.ndarray,
    heatmap_b: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
    """Compute the per-pixel disagreement map and contested-region statistics.

    Both inputs are independently min-max normalized to ``[0, 1]``. The
    disagreement map is the absolute difference of the normalized maps. The
    "contested" region is the top-20%-mass of the *combined* activation
    (``norm_a + norm_b``), i.e. pixels at or above the 80th percentile of that
    combined map. Region statistics for each agent are reported within that
    mask.

    Args:
        heatmap_a: Agent A's saliency map (any shape; typically ``224x224``).
        heatmap_b: Agent B's saliency map, same shape as ``heatmap_a``.

    Returns:
        A tuple ``(m_delta, region_stats_a, region_stats_b)`` where ``m_delta``
        is the ``float32`` absolute-difference map and each ``region_stats`` dict
        contains ``"mean"``, ``"std"`` and ``"max"`` floats for that agent inside
        the contested mask.
    """
    norm_a = _min_max_normalize(heatmap_a)
    norm_b = _min_max_normalize(heatmap_b)

    m_delta = np.abs(norm_a - norm_b).astype(np.float32)

    combined = norm_a + norm_b
    threshold = float(np.percentile(combined, 80.0))
    mask = combined >= threshold

    # Guard against an empty / degenerate mask (e.g. a constant combined map):
    # fall back to selecting the entire frame so statistics remain meaningful.
    if not bool(mask.any()):
        mask = np.ones_like(combined, dtype=bool)

    region_stats_a = _region_stats(norm_a, mask)
    region_stats_b = _region_stats(norm_b, mask)

    return (
        np.ascontiguousarray(m_delta, dtype=np.float32),
        region_stats_a,
        region_stats_b,
    )


def extract_bbox(
    disagreement_map: np.ndarray,
    top_k_percent: float = 0.20,
) -> BoundingBox:
    """Extract a bounding box around the most contested pixels.

    Pixels above the ``(1 - top_k_percent)`` quantile of ``disagreement_map`` are
    treated as contested; the minimal axis-aligned rectangle enclosing them is
    returned. If no pixel exceeds the quantile (e.g. a constant map), a
    full-frame box is returned.

    Args:
        disagreement_map: A 2-D disagreement map (``M_delta``).
        top_k_percent: Fraction of the highest-disagreement pixels to enclose
            (default ``0.20`` => top 20%).

    Returns:
        A :class:`~core.models.BoundingBox` with integer ``x1, y1, x2, y2`` where
        ``x`` is the column and ``y`` is the row. Coordinates are inclusive of the
        extreme contested pixels.
    """
    arr = np.asarray(disagreement_map, dtype=np.float32)
    rows, cols = arr.shape

    quantile = float(np.clip(1.0 - top_k_percent, 0.0, 1.0))
    threshold = float(np.quantile(arr, quantile))
    mask = arr > threshold

    if not bool(mask.any()):
        # No pixel strictly exceeds the quantile: return a full-frame box.
        return BoundingBox(x1=0, y1=0, x2=int(cols - 1), y2=int(rows - 1))

    row_indices, col_indices = np.where(mask)
    y1 = int(row_indices.min())
    y2 = int(row_indices.max())
    x1 = int(col_indices.min())
    x2 = int(col_indices.max())

    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
