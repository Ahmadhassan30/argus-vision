"""Debate trigger evaluation for the Argus Vision pipeline.

The trigger decides whether the two classifier agents disagree enough — or are
individually uncertain enough — to warrant an adversarial debate. It quantifies
inter-agent disagreement with the (squared) Jensen-Shannon divergence and the
per-agent uncertainty with the Shannon entropy of each predictive distribution.

All probability inputs are coerced to ``numpy`` arrays laid out in the canonical
ISIC-8 class order before any maths is performed, so callers may pass either a
``{class_code: probability}`` mapping or a pre-ordered array interchangeably.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy.spatial.distance import jensenshannon

from core.models import TriggerResult

# ISIC-8 class names in their canonical (index 0..7) order. This ordering is the
# single source of truth used to flatten probability dictionaries into vectors.
CLASS_NAMES: list[str] = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

NUM_CLASSES: int = 8

# Numerical floor added inside the logarithm to keep entropy finite when a
# probability is exactly zero.
_EPS: float = 1e-12

ProbInput = Union["dict[str, float]", np.ndarray, "list[float]"]


def coerce_to_ordered_array(probs: ProbInput) -> np.ndarray:
    """Convert a probability input into an ordered ISIC-8 ``numpy`` array.

    A mapping is read in the canonical :data:`CLASS_NAMES` order (missing keys
    default to ``0.0``); an array-like is converted as-is and validated to have
    exactly :data:`NUM_CLASSES` entries.

    Args:
        probs: Either a ``{class_code: probability}`` mapping or an array-like
            of length :data:`NUM_CLASSES` already in canonical order.

    Returns:
        A ``float64`` array of shape ``(NUM_CLASSES,)`` in canonical class
        order.

    Raises:
        ValueError: If an array-like input does not have exactly
            :data:`NUM_CLASSES` elements.
    """
    if isinstance(probs, dict):
        return np.array(
            [float(probs.get(name, 0.0)) for name in CLASS_NAMES],
            dtype=np.float64,
        )

    array = np.asarray(probs, dtype=np.float64).ravel()
    if array.shape[0] != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} probabilities, got {array.shape[0]}."
        )
    return array


def shannon_entropy(probs: np.ndarray) -> float:
    """Compute the base-2 Shannon entropy of a probability vector.

    Args:
        probs: A probability vector of shape ``(NUM_CLASSES,)``. It need not be
            perfectly normalised; an :data:`_EPS` floor guards the logarithm.

    Returns:
        The Shannon entropy in bits as a Python ``float``.
    """
    p = np.clip(probs, 0.0, None)
    entropy = -float(np.sum(p * np.log2(p + _EPS)))
    return entropy


def evaluate_trigger(
    probs_a: ProbInput,
    probs_b: ProbInput,
    tau_js: float,
    tau_entropy: float,
) -> TriggerResult:
    """Decide whether an adversarial debate should be triggered.

    The decision fires when the two agents' predictive distributions diverge
    beyond ``tau_js`` (squared Jensen-Shannon divergence) **or** when either
    agent's distribution is more uncertain than ``tau_entropy`` (Shannon
    entropy in bits).

    Args:
        probs_a: Agent A predictive distribution, as a class-code mapping or a
            canonically ordered array-like.
        probs_b: Agent B predictive distribution, in the same accepted formats.
        tau_js: Jensen-Shannon divergence threshold above which disagreement is
            considered significant.
        tau_entropy: Entropy threshold (bits) above which an agent is considered
            too uncertain to trust on its own.

    Returns:
        A :class:`~core.models.TriggerResult` carrying the firing decision and
        every intermediate metric used to reach it.
    """
    pa = coerce_to_ordered_array(probs_a)
    pb = coerce_to_ordered_array(probs_b)

    # ``jensenshannon`` returns the JS *distance* (the square root of the JS
    # divergence). Squaring it recovers the divergence in [0, 1] for base=2.
    js_distance = jensenshannon(pa, pb, base=2)
    js_divergence = float(js_distance) ** 2
    if not np.isfinite(js_divergence):
        # A NaN/inf can occur if both vectors are all-zero; treat as agreement.
        js_divergence = 0.0

    entropy_a = shannon_entropy(pa)
    entropy_b = shannon_entropy(pb)

    fired = (js_divergence > tau_js) or (max(entropy_a, entropy_b) > tau_entropy)

    return TriggerResult(
        fired=fired,
        js_divergence=js_divergence,
        entropy_a=entropy_a,
        entropy_b=entropy_b,
        threshold_js=tau_js,
        threshold_entropy=tau_entropy,
    )
