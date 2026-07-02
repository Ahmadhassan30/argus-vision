"""Unit tests for the Phase 10 dermoscopy input gate.

Tests cover both stages:
  * Stage 1 — :func:`is_plausible_dermoscopy` (heuristic checks).
  * Stage 2 — :class:`DermoscopyClassifierGate` (mocked to avoid downloading
    real ``timm`` weights in the test environment).

Requires Pillow and numpy (both already in requirements.txt).
"""

from __future__ import annotations

import io
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Make ``ml.dermoscopy_gate`` importable regardless of pytest's invocation dir.
_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from ml.dermoscopy_gate import is_plausible_dermoscopy, INVALID_CLASS_IDS



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeSettings:
    """Mimics the subset of Settings used by ``is_plausible_dermoscopy``."""

    DERMOSCOPY_GATE_ENABLED: bool = True
    DERMOSCOPY_ASPECT_RATIO_MAX: float = 2.5
    DERMOSCOPY_MIN_DIMENSION_PX: int = 50
    DERMOSCOPY_CHANNEL_STD_MIN: float = 8.0
    DERMOSCOPY_CLASSIFIER_CONFIDENCE_THRESHOLD: float = 0.60
    DERMOSCOPY_GATE_MODEL: str = "mobilenetv3_small_100"


def _make_image(
    width: int = 224,
    height: int = 224,
    colour: tuple[int, int, int] = (160, 120, 100),
    noise: bool = True,
) -> Image.Image:
    """Create a synthetic test image.

    When *noise* is True the image gets random pixel variation so it passes the
    channel-variance check (mimics a real dermoscopy photo).  When False a solid
    flat colour is returned (should be caught by the variance check).
    """
    arr = np.full((height, width, 3), colour, dtype=np.uint8)
    if noise:
        rng = np.random.default_rng(42)
        arr = np.clip(arr.astype(np.int16) + rng.integers(-30, 30, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


# ---------------------------------------------------------------------------
# Stage 1: heuristic checks
# ---------------------------------------------------------------------------

class TestHeuristicPreFilter:
    """Tests for ``is_plausible_dermoscopy``."""

    settings = _FakeSettings()

    def test_valid_dermoscopy_passes(self) -> None:
        """A 224×224 image with realistic variation passes all checks."""
        img = _make_image(224, 224, noise=True)
        passed, reason = is_plausible_dermoscopy(img, self.settings)
        assert passed is True
        assert reason == ""

    def test_too_small_rejected(self) -> None:
        """An image below the minimum dimension is rejected."""
        img = _make_image(30, 30, noise=True)
        passed, reason = is_plausible_dermoscopy(img, self.settings)
        assert passed is False
        assert "too small" in reason.lower()

    def test_extreme_aspect_ratio_rejected(self) -> None:
        """A very wide panoramic image is rejected."""
        img = _make_image(800, 100, noise=True)
        passed, reason = is_plausible_dermoscopy(img, self.settings)
        assert passed is False
        assert "aspect ratio" in reason.lower()

    def test_solid_white_rejected(self) -> None:
        """A solid white image is caught by the channel-variance check."""
        img = _make_image(224, 224, colour=(255, 255, 255), noise=False)
        passed, reason = is_plausible_dermoscopy(img, self.settings)
        assert passed is False
        assert "blank" in reason.lower() or "uniform" in reason.lower()

    def test_solid_black_rejected(self) -> None:
        """A solid black image is caught by the channel-variance check."""
        img = _make_image(224, 224, colour=(0, 0, 0), noise=False)
        passed, reason = is_plausible_dermoscopy(img, self.settings)
        assert passed is False
        assert "blank" in reason.lower() or "uniform" in reason.lower()

    def test_borderline_aspect_ratio_passes(self) -> None:
        """An aspect ratio just under the limit passes."""
        # 2.4:1 is below the 2.5 threshold
        img = _make_image(480, 200, noise=True)
        passed, reason = is_plausible_dermoscopy(img, self.settings)
        assert passed is True

    def test_borderline_aspect_ratio_fails(self) -> None:
        """An aspect ratio just over the limit is rejected."""
        # 3:1 exceeds the 2.5 threshold
        img = _make_image(600, 200, noise=True)
        passed, reason = is_plausible_dermoscopy(img, self.settings)
        assert passed is False
        assert "aspect ratio" in reason.lower()


# ---------------------------------------------------------------------------
# Stage 2: classifier gate (mocked — no real model weights needed)
# ---------------------------------------------------------------------------

_has_torch = True
try:
    import torch as _torch
except ImportError:
    _has_torch = False


@pytest.mark.skipif(not _has_torch, reason="torch not installed (runs in Docker)")
class TestClassifierGate:
    """Tests for ``DermoscopyClassifierGate`` using mocked timm model."""

    def test_rejection_on_invalid_class(self) -> None:
        """Gate rejects when top-1 class is in the invalid set with high confidence."""
        import torch

        # Pick a known invalid class index (index 0 = tench, a fish).
        invalid_idx = 0
        assert invalid_idx in INVALID_CLASS_IDS

        # Build a fake logits tensor that puts all probability mass on the fish class.
        fake_logits = torch.zeros(1, 1000)
        fake_logits[0, invalid_idx] = 10.0  # will become ~1.0 after softmax

        with patch("ml.dermoscopy_gate.DermoscopyClassifierGate.__init__", return_value=None):
            from ml.dermoscopy_gate import DermoscopyClassifierGate

            gate = DermoscopyClassifierGate.__new__(DermoscopyClassifierGate)
            gate._device = torch.device("cpu")
            gate._model = MagicMock(return_value=fake_logits)
            gate._transform = MagicMock(return_value=torch.randn(3, 224, 224))
            gate._imagenet_labels = {0: "tench"}

        img = _make_image(224, 224, noise=True)
        passed, reason = gate.is_dermoscopy(img, confidence_threshold=0.60)
        assert passed is False
        assert "tench" in reason.lower()

    def test_allowed_when_confidence_below_threshold(self) -> None:
        """Gate allows image when confidence is below threshold."""
        import torch

        # Spread probability evenly — no class will be above 0.60
        fake_logits = torch.ones(1, 1000)

        with patch("ml.dermoscopy_gate.DermoscopyClassifierGate.__init__", return_value=None):
            from ml.dermoscopy_gate import DermoscopyClassifierGate

            gate = DermoscopyClassifierGate.__new__(DermoscopyClassifierGate)
            gate._device = torch.device("cpu")
            gate._model = MagicMock(return_value=fake_logits)
            gate._transform = MagicMock(return_value=torch.randn(3, 224, 224))
            gate._imagenet_labels = {}

        img = _make_image(224, 224, noise=True)
        passed, reason = gate.is_dermoscopy(img, confidence_threshold=0.60)
        assert passed is True
        assert reason == ""

    def test_allowed_when_class_not_in_invalid_set(self) -> None:
        """Gate allows image when top-1 class is NOT in INVALID_CLASS_IDS."""
        import torch

        # Pick a class that is NOT in the invalid set (e.g., 551 = "Band-Aid",
        # deliberately excluded from INVALID_CLASS_IDS).
        safe_idx = 551
        assert safe_idx not in INVALID_CLASS_IDS

        fake_logits = torch.zeros(1, 1000)
        fake_logits[0, safe_idx] = 10.0

        with patch("ml.dermoscopy_gate.DermoscopyClassifierGate.__init__", return_value=None):
            from ml.dermoscopy_gate import DermoscopyClassifierGate

            gate = DermoscopyClassifierGate.__new__(DermoscopyClassifierGate)
            gate._device = torch.device("cpu")
            gate._model = MagicMock(return_value=fake_logits)
            gate._transform = MagicMock(return_value=torch.randn(3, 224, 224))
            gate._imagenet_labels = {551: "Band-Aid"}

        img = _make_image(224, 224, noise=True)
        passed, reason = gate.is_dermoscopy(img, confidence_threshold=0.60)
        assert passed is True
        assert reason == ""


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-v"]))
