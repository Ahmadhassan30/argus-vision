"""Dermoscopy input validity gate (Phase 10).

Two-stage pre-filter that runs **before** any Agent A / Agent B inference to
reject images that are clearly not dermoscopic photographs.

Stage 1 — Fast heuristic pre-filter (no model)
    :func:`is_plausible_dermoscopy` checks aspect ratio, minimum resolution, and
    per-channel colour variance.  These are O(1) / O(pixels) and add < 1 ms.

Stage 2 — Lightweight ImageNet classifier gate
    :class:`DermoscopyClassifierGate` loads a MobileNetV3-Small from ``timm``,
    runs a single forward pass, and checks whether the top-1 predicted ImageNet
    class falls into a curated set of *obviously non-medical* categories
    (animals, vehicles, food, electronics, furniture, etc.).  If the confidence
    exceeds a configurable threshold the image is rejected.

Known Limitations
-----------------
This gate catches obviously non-medical inputs (pets, screenshots, food,
vehicles, etc.) but does **not** reject medical-but-non-dermoscopy images
such as chest X-rays, histopathology slides, fundus photographs, or
ultrasound frames.  Those will pass through to the agents and produce
nonsensical skin-lesion predictions.

The ImageNet class taxonomy has no "dermoscopy" or "skin lesion" label, so
the classifier can only flag images it *does* recognise as non-skin.  Any
image that the classifier is uncertain about (low confidence for all invalid
classes) is allowed through — a deliberate "fail open" design to avoid
blocking legitimate clinical uploads.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from core.config import Settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 — fast heuristic checks
# ---------------------------------------------------------------------------

def is_plausible_dermoscopy(
    image: Image.Image,
    settings: "Settings",
) -> tuple[bool, str]:
    """Run cheap heuristic checks on an already-opened PIL image.

    Returns ``(True, "")`` when the image passes all checks, or
    ``(False, reason)`` with a human-readable rejection reason on failure.

    Checks performed (in order):
    1. **Minimum dimension** — both width and height must be ≥
       ``settings.DERMOSCOPY_MIN_DIMENSION_PX``.
    2. **Aspect ratio** — ``max(w, h) / min(w, h)`` must be ≤
       ``settings.DERMOSCOPY_ASPECT_RATIO_MAX``.  Dermoscopy images are
       approximately square; panoramic / ultra-wide images are rejected.
    3. **Channel variance** — if *all three* RGB channel standard deviations
       are below ``settings.DERMOSCOPY_CHANNEL_STD_MIN`` the image is
       essentially a solid colour or blank.

    Args:
        image: An RGB PIL ``Image`` (already converted).
        settings: Application :class:`~core.config.Settings` providing the
            threshold values.

    Returns:
        A ``(passed, reason)`` tuple.
    """
    w, h = image.size

    # Check 1 — minimum dimension
    if w < settings.DERMOSCOPY_MIN_DIMENSION_PX or h < settings.DERMOSCOPY_MIN_DIMENSION_PX:
        return (
            False,
            f"Image too small ({w}×{h} px). Both dimensions must be at "
            f"least {settings.DERMOSCOPY_MIN_DIMENSION_PX} px.",
        )

    # Check 2 — aspect ratio
    long_side = max(w, h)
    short_side = min(w, h)
    ratio = long_side / short_side
    if ratio > settings.DERMOSCOPY_ASPECT_RATIO_MAX:
        return (
            False,
            f"Aspect ratio {ratio:.2f}:1 exceeds the maximum of "
            f"{settings.DERMOSCOPY_ASPECT_RATIO_MAX:.1f}:1. Dermoscopy images "
            f"are approximately square.",
        )

    # Check 3 — channel variance (solid colour / blank detection)
    # Downsample to ≤ 128×128 for speed — std doesn't need full resolution.
    thumb = image.copy()
    thumb.thumbnail((128, 128), Image.Resampling.NEAREST)
    arr = np.asarray(thumb, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[2] >= 3:
        stds = [float(arr[:, :, c].std()) for c in range(3)]
        if all(s < settings.DERMOSCOPY_CHANNEL_STD_MIN for s in stds):
            return (
                False,
                f"Image appears blank or uniform (channel σ = "
                f"[{stds[0]:.1f}, {stds[1]:.1f}, {stds[2]:.1f}], all below "
                f"{settings.DERMOSCOPY_CHANNEL_STD_MIN:.1f}).",
            )

    return (True, "")


# ---------------------------------------------------------------------------
# Stage 2 — lightweight ImageNet classifier gate
# ---------------------------------------------------------------------------

# Curated set of ImageNet-1K class indices that are *obviously* not
# dermoscopy.  Grouped by semantic category for readability.  Indices follow
# the canonical torchvision / timm ordering (synset-alphabetical, 0-999).
#
# Only high-confidence hits in these classes trigger rejection.  Classes that
# are ambiguous (e.g. "Band-Aid" at idx 551) are intentionally excluded to
# avoid false positives on clinical images.
INVALID_CLASS_IDS: frozenset[int] = frozenset(
    # ── Animals ──────────────────────────────────────────────────────────
    list(range(0, 398))       # 0-397: animals (birds, fish, reptiles, mammals …)
    # ── Vehicles / transport ────────────────────────────────────────────
    + [
        407,  # ambulance
        436,  # beach wagon
        444,  # bicycle
        468,  # cab / taxi
        479,  # car wheel
        511,  # convertible
        555,  # fire engine
        569,  # freight car
        573,  # garbage truck
        586,  # half-track
        609,  # jeep
        627,  # limousine
        654,  # minivan
        656,  # model T
        665,  # motor scooter
        670,  # moving van
        675,  # moped / ox-cart
        705,  # passenger car
        717,  # pickup truck
        734,  # police van
        751,  # racer / race car
        779,  # school bus
        817,  # sports car
        864,  # tow truck
        867,  # trailer truck
        874,  # trolleybus
    ]
    # ── Electronics / office ────────────────────────────────────────────
    + [
        527,  # desktop computer
        528,  # dial telephone
        590,  # hand-held computer (PDA)
        620,  # laptop
        621,  # lawn mower
        664,  # monitor
        681,  # notebook / laptop
        710,  # pencil box
        720,  # pill bottle (ambiguous? no — pills ≠ skin lesion)
        745,  # printer
        752,  # radio
        762,  # remote control
        782,  # screen / CRT
        851,  # television
        852,  # tennis ball
    ]
    # ── Food / kitchen ──────────────────────────────────────────────────
    + [
        924,  # guacamole
        925,  # gyoza
        926,  # hair slide (not food, but non-medical)
        927,  # hamburger
        928,  # hammer
        929,  # hamper
        930,  # harmonica
        932,  # hash browns (custom — not real idx, but range keeps it safe)
        935,  # hot dog
        939,  # ice cream
        940,  # ice lolly
        956,  # mashed potato
        959,  # meat loaf
        963,  # pizza
        965,  # pretzel
        966,  # pomegranate (fruit)
        969,  # espresso
    ]
    # ── Buildings / scenery ─────────────────────────────────────────────
    + [
        449,  # bookshop
        483,  # castle
        497,  # church
        508,  # confectionery
        536,  # dock
        648,  # megalith
        663,  # monastery
        669,  # mosque
        698,  # palace
        725,  # planetarium
        838,  # stupa
        878,  # upright
    ]
    # ── Furniture / household ───────────────────────────────────────────
    + [
        423,  # barber chair
        559,  # folding chair
        765,  # rocking chair
        831,  # studio couch
        857,  # throne
        423,  # barber chair (dupe, harmless in frozenset)
    ]
)


# TODO(phase-N): Replace the ImageNet-based heuristic gate with a purpose-trained
# binary classifier ("dermoscopy vs. everything-else") fine-tuned on a curated
# dataset of dermoscopy positives + hard negatives (X-ray, fundus, histopath,
# everyday photos).  This is the production upgrade path for closing the
# medical-but-non-dermoscopy gap described in the module docstring.


class DermoscopyClassifierGate:
    """Lightweight MobileNetV3-based gate to reject non-dermoscopy images.

    Loads a ``timm`` MobileNetV3-Small pretrained on ImageNet-1K once at
    construction time.  The :meth:`is_dermoscopy` method runs a single forward
    pass and checks whether the top-1 predicted class belongs to
    :data:`INVALID_CLASS_IDS`.

    This gate is deliberately "fail open": if the model is uncertain (top-1
    confidence below the threshold, or the predicted class is *not* in the
    invalid set) the image is allowed through.  This avoids rejecting
    legitimate but unusual dermoscopy images.
    """

    def __init__(self, model_name: str = "mobilenetv3_small_100") -> None:
        """Load the classifier once.

        Args:
            model_name: A ``timm`` model identifier.  Must be an ImageNet-1K
                pretrained model returning 1000-class logits.
        """
        import timm
        import torch

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = timm.create_model(model_name, pretrained=True).eval().to(self._device)

        # Grab the timm data config for this model so we use the correct
        # preprocessing (resize, crop, mean/std).
        data_cfg = timm.data.resolve_model_data_config(self._model)
        self._transform = timm.data.create_transform(**data_cfg, is_training=False)

        # Load the ImageNet-1K class index → label mapping that ships with timm.
        self._imagenet_labels: dict[int, str] = {}
        try:
            # timm ≥ 0.9 exposes an id-to-label dict via the model config or
            # via ``timm.data.imagenet_info``.  Fall back gracefully.
            from timm.data import ImageNetInfo
            info = ImageNetInfo()
            self._imagenet_labels = {i: info.label_names[i] for i in range(len(info.label_names))}
        except Exception:
            # Older timm — label names aren't critical, we'll use indices.
            pass

        logger.info(
            "DermoscopyClassifierGate loaded (%s on %s, %d invalid class IDs).",
            model_name,
            self._device,
            len(INVALID_CLASS_IDS),
        )

    def is_dermoscopy(
        self,
        image: Image.Image,
        confidence_threshold: float = 0.60,
    ) -> tuple[bool, str]:
        """Classify *image* and reject if it matches an obviously invalid class.

        Args:
            image: An RGB PIL ``Image``.
            confidence_threshold: Minimum softmax confidence for the top-1
                class to trigger a rejection.  Lower values are stricter.

        Returns:
            ``(True, "")`` if the image is allowed (either it does not match
            any invalid class, or the confidence is below the threshold).
            ``(False, reason)`` if rejected.
        """
        import torch

        tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)

        probs = torch.softmax(logits, dim=-1).squeeze(0)
        top_conf, top_idx = probs.max(dim=0)
        top_conf = float(top_conf)
        top_idx = int(top_idx)

        if top_idx in INVALID_CLASS_IDS and top_conf >= confidence_threshold:
            label = self._imagenet_labels.get(top_idx, f"class-{top_idx}")
            reason = (
                f"Detected: '{label}' (confidence {top_conf:.0%}). "
                f"This does not appear to be a dermoscopy image."
            )
            logger.info("Dermoscopy gate rejected image: %s", reason)
            return (False, reason)

        return (True, "")
