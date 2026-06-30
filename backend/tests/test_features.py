"""Closed-form unit tests for the 23-d consensus feature extractor.

These pin the exact numerical contract of
``ml.debate.features.extract_consensus_features`` — the serving-path feature
layout that must stay byte-for-byte identical to the training/eval code. Several
Phase 0-8 bugs were silent feature drift (wrong JS log base, wrong dim, NaNs from
all-zero attention), so each assertion below is a regression guard against one of
those failure modes.

The values are hand-derived, not snapshotted: a uniform distribution has base-2
entropy of exactly ``log2(8) == 3.0`` bits, two disjoint one-hots have a natural-
log JS *divergence* of ``ln(2)`` (this is what distinguishes features.py's base-e
JS from trigger.py's base-2 JS), and an identical pair of binary attention maps
has IoU and per-map base-2 entropy of 1.0.

Requires numpy + scipy installed to run (not available in the audit env).
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Make ``ml.debate.features`` importable regardless of pytest's invocation dir.
_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from ml.debate.features import extract_consensus_features  # noqa: E402


def _onehot(index: int, n: int = 8) -> np.ndarray:
    """Return a length-``n`` one-hot vector with a 1.0 at ``index``."""
    vec = np.zeros(n, dtype=np.float64)
    vec[index] = 1.0
    return vec


def test_uniform_pair_no_attention() -> None:
    """Two identical uniform distributions with no attention maps.

    Pins: probs copied through verbatim (0.125), zero JS divergence, base-2
    entropy of exactly 3.0 bits per agent, zero max-prob delta, and zero
    attention features (the legitimate no-attention fast path).
    """
    feat = extract_consensus_features([1 / 8] * 8, [1 / 8] * 8, None, None)

    assert feat.shape == (23,)
    assert feat.dtype == np.float32
    assert not np.any(np.isnan(feat))
    assert not np.any(np.isinf(feat))

    # prob_a (feat[:8]) and prob_b (feat[8:16]) pass through unchanged.
    assert np.all(feat[:16] == np.float32(0.125))

    # JS divergence of identical distributions is 0.
    assert abs(float(feat[16])) < 1e-6

    # Base-2 entropy of a uniform 8-vector is log2(8) == 3.0 (exact after the
    # float32 cast snaps the float64 rounding error back to 3.0).
    assert feat[17] == 3.0
    assert feat[18] == 3.0

    # No probability differs between the two agents.
    assert feat[19] == 0.0

    # No attention maps -> the three attention features are exactly 0.
    assert np.all(feat[20:23] == 0.0)


def test_disjoint_onehots_natural_log_js() -> None:
    """Disjoint one-hots pin the natural-log JS base used in features.py.

    feat[16] must equal ln(2) ~= 0.693, NOT 1.0 (which is the base-2 answer used
    by trigger.py). This is the single assertion that catches a JS log-base
    regression between the two modules.
    """
    feat = extract_consensus_features(_onehot(0), _onehot(1), None, None)

    assert abs(float(feat[16]) - float(np.log(2))) < 1e-3  # ln(2) ~= 0.693
    assert abs(float(feat[19]) - 1.0) < 1e-3               # max prob delta ~= 1
    assert abs(float(feat[17])) < 1e-3                      # near-zero entropy
    assert abs(float(feat[18])) < 1e-3


def test_identical_attention_maps() -> None:
    """Identical binary attention maps -> IoU and per-map entropy of ~1.0."""
    attn = [[0, 1], [1, 0]]
    feat = extract_consensus_features([1 / 8] * 8, [1 / 8] * 8, attn, attn)

    assert abs(float(feat[20]) - 1.0) < 1e-6   # attn IoU
    assert abs(float(feat[21]) - 1.0) < 1e-6   # attn entropy A (base 2)
    assert abs(float(feat[22]) - 1.0) < 1e-6   # attn entropy B (base 2)


def test_js_divergence_is_symmetric() -> None:
    """JS divergence is order-independent: extract(a,b)[16] == extract(b,a)[16]."""
    a = [0.6, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    b = [0.1, 0.5, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05]

    assert extract_consensus_features(a, b)[16] == extract_consensus_features(b, a)[16]


if __name__ == "__main__":
    test_uniform_pair_no_attention()
    test_disjoint_onehots_natural_log_js()
    test_identical_attention_maps()
    test_js_divergence_is_symmetric()
    print("[PASS] extract_consensus_features closed-form contract.")
