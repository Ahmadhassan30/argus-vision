"""Unit tests for the debate-trigger maths in ``ml.debate.trigger``.

Covers the three public helpers — ``coerce_to_ordered_array`` (dict/array
normalisation into canonical ISIC-8 order), ``shannon_entropy`` (base-2), and
``evaluate_trigger`` (the JS-OR-entropy firing decision). The assertions pin the
base-2 JS divergence (distinct from features.py's natural-log JS), the entropy
OR-branch, threshold echoing, and the all-zero NaN guard — each a Phase 0-8
regression that produced wrong firing decisions or crashes.

Requires numpy + scipy installed to run (not available in the audit env).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Make ``ml.debate.trigger`` / ``core.models`` importable regardless of cwd.
_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from ml.debate.trigger import (  # noqa: E402
    coerce_to_ordered_array,
    evaluate_trigger,
    shannon_entropy,
)


def _onehot(index: int, n: int = 8) -> np.ndarray:
    """Return a length-``n`` one-hot vector with a 1.0 at ``index``."""
    vec = np.zeros(n, dtype=np.float64)
    vec[index] = 1.0
    return vec


def test_coerce_dict_orders_canonically() -> None:
    """A {class: prob} mapping is laid out in canonical ISIC-8 order."""
    ordered = coerce_to_ordered_array({"MEL": 0.5, "NV": 0.5})
    assert np.array_equal(ordered, [0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_coerce_wrong_length_raises() -> None:
    """An array-like that is not exactly NUM_CLASSES long is rejected."""
    with pytest.raises(ValueError):
        coerce_to_ordered_array([0.0] * 7)


def test_shannon_entropy_uniform_and_onehot() -> None:
    """Base-2 entropy: 3.0 bits for uniform-8, ~0.0 for a one-hot."""
    assert abs(shannon_entropy(np.full(8, 1 / 8)) - 3.0) < 1e-6
    assert abs(shannon_entropy(_onehot(0))) < 1e-6


def test_evaluate_trigger_disjoint_fires_on_js() -> None:
    """Disjoint one-hots: fires on the JS branch with base-2 divergence ~= 1.0."""
    result = evaluate_trigger(_onehot(0), _onehot(1), 0.25, 0.8)
    assert result.fired is True
    assert abs(result.js_divergence - 1.0) < 1e-6  # base=2 disjoint
    # Thresholds are echoed back verbatim.
    assert result.threshold_js == 0.25
    assert result.threshold_entropy == 0.8


def test_evaluate_trigger_identical_onehot_does_not_fire() -> None:
    """Identical confident predictions: no divergence, no firing."""
    result = evaluate_trigger(_onehot(0), _onehot(0), 0.25, 0.8)
    assert result.fired is False
    assert abs(result.js_divergence) < 1e-6


def test_evaluate_trigger_fires_via_entropy_branch() -> None:
    """Agreeing-but-uncertain agents fire via the entropy OR-branch, not JS."""
    result = evaluate_trigger([1 / 8] * 8, [1 / 8] * 8, 0.25, 0.8)
    assert result.fired is True
    assert abs(result.js_divergence) < 1e-6     # JS branch did NOT fire
    assert abs(result.entropy_a - 3.0) < 1e-6   # entropy branch did


def test_evaluate_trigger_all_zero_is_nan_guarded() -> None:
    """All-zero inputs must not raise and must be treated as agreement."""
    result = evaluate_trigger([0.0] * 8, [0.0] * 8, 0.25, 0.8)
    assert result.js_divergence == 0.0
    assert result.fired is False


if __name__ == "__main__":
    test_coerce_dict_orders_canonically()
    test_coerce_wrong_length_raises()
    test_shannon_entropy_uniform_and_onehot()
    test_evaluate_trigger_disjoint_fires_on_js()
    test_evaluate_trigger_identical_onehot_does_not_fire()
    test_evaluate_trigger_fires_via_entropy_branch()
    test_evaluate_trigger_all_zero_is_nan_guarded()
    print("[PASS] evaluate_trigger / shannon_entropy / coerce contract.")
