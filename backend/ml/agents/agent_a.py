"""Agent A: EfficientNet-B4 classifier for ISIC-8 dermoscopic image classification.

This module defines :class:`AgentA`, the first participant in the Argus Vision
adversarial debate. It wraps a ``timm`` EfficientNet-B4 backbone with an 8-class
head and exposes a clean inference API returning a
:class:`core.models.ClassificationResult`.
"""

from __future__ import annotations

import logging
import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.exceptions import ModelNotLoadedError
from core.models import ClassificationResult

logger = logging.getLogger(__name__)

# ISIC-8 class names in their canonical (index 0..7) order.
CLASS_NAMES: list[str] = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# Number of output classes for the classification head.
NUM_CLASSES: int = 8

# The timm model identifier for the EfficientNet-B4 backbone.
MODEL_NAME: str = "efficientnet_b4"


class AgentA:
    """EfficientNet-B4 based classifier (Agent A) for the Argus Vision debate.

    Agent A loads an EfficientNet-B4 model from ``timm`` with an 8-class head.
    A trained checkpoint is loaded when available; otherwise, if
    ``pretrained_fallback`` is enabled, an ImageNet-pretrained backbone with a
    freshly initialized classification head is used (predictions are then not
    clinically meaningful). If neither path is possible a
    :class:`core.exceptions.ModelNotLoadedError` is raised.

    Attributes:
        agent_id: Stable identifier for this agent, always ``"A"``.
        CLASS_NAMES: The ISIC-8 class names in canonical order.
        device: The resolved torch device string (``"cuda"`` or ``"cpu"``).
        model: The underlying ``nn.Module`` in evaluation mode.
    """

    agent_id: str = "A"
    CLASS_NAMES: list[str] = CLASS_NAMES

    def __init__(
        self,
        checkpoint_path: str | None,
        pretrained_fallback: bool,
        device: str | None = None,
    ) -> None:
        """Initialize Agent A and load its weights.

        Args:
            checkpoint_path: Path to a fine-tuned ``.pth`` state dict. If the
                file exists on disk it is loaded into a randomly initialized
                (non-pretrained) backbone with ``strict=False``.
            pretrained_fallback: When ``checkpoint_path`` does not exist on
                disk, controls whether to fall back to an ImageNet-pretrained
                backbone with a random 8-class head (``True``) or to raise
                :class:`core.exceptions.ModelNotLoadedError` (``False``).
            device: Optional torch device string. When ``None`` the device is
                resolved automatically to ``"cuda"`` if available else
                ``"cpu"``.

        Raises:
            ModelNotLoadedError: If no checkpoint exists and
                ``pretrained_fallback`` is ``False``.
        """
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            logger.info(
                "Agent A: loading checkpoint from '%s' onto device '%s'.",
                checkpoint_path,
                self.device,
            )
            self.model: nn.Module = timm.create_model(
                MODEL_NAME,
                pretrained=False,
                num_classes=NUM_CLASSES,
            )
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            load_result = self.model.load_state_dict(state_dict, strict=False)
            if load_result.missing_keys:
                logger.warning(
                    "Agent A: %d missing key(s) when loading checkpoint: %s",
                    len(load_result.missing_keys),
                    load_result.missing_keys,
                )
            if load_result.unexpected_keys:
                logger.warning(
                    "Agent A: %d unexpected key(s) when loading checkpoint: %s",
                    len(load_result.unexpected_keys),
                    load_result.unexpected_keys,
                )
            # A clean fine-tuned checkpoint must match the architecture exactly; a
            # non-empty missing/unexpected list means a silent train/serve model mismatch.
            assert not load_result.missing_keys and not load_result.unexpected_keys, (
                f"Agent A checkpoint key mismatch — missing={load_result.missing_keys}, "
                f"unexpected={load_result.unexpected_keys}. Refusing to serve a mismatched model."
            )
        elif pretrained_fallback:
            logger.error(
                "Agent A: no checkpoint found at '%s'. Falling back to an "
                "ImageNet-pretrained EfficientNet-B4 backbone with a "
                "randomly-initialized 8-class head. Predictions are NOT "
                "clinically meaningful.",
                checkpoint_path,
            )
            self.model = timm.create_model(
                MODEL_NAME,
                pretrained=True,
                num_classes=NUM_CLASSES,
            )
        else:
            raise ModelNotLoadedError(
                f"Agent A: no checkpoint found at '{checkpoint_path}' and "
                "pretrained_fallback is disabled."
            )

        self.model.eval()
        self.model.to(self.device)

    def predict(self, tensor: torch.Tensor) -> ClassificationResult:
        """Run a forward pass and return a structured classification result.

        Args:
            tensor: A pre-processed input batch of shape ``(N, 3, 224, 224)``
                (typically ``N == 1``). The tensor is moved to the agent's
                device before inference.

        Returns:
            A :class:`core.models.ClassificationResult` with the top predicted
            class name, its confidence (maximum softmax probability), and the
            full per-class probability mapping.
        """
        tensor = tensor.to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)

        # Use the first item in the batch for the structured result.
        prob_row = probs[0]
        top_index = int(torch.argmax(prob_row).item())

        probabilities: dict[str, float] = {
            class_name: float(prob_row[idx].item())
            for idx, class_name in enumerate(self.CLASS_NAMES)
        }

        return ClassificationResult(
            pred_class=self.CLASS_NAMES[top_index],
            confidence=float(prob_row[top_index].item()),
            probabilities=probabilities,
        )

    def get_model(self) -> nn.Module:
        """Return the raw underlying model.

        This is required by GradCAM, which needs direct access to the
        convolutional backbone to register hooks on its feature layers.

        Returns:
            The wrapped EfficientNet-B4 :class:`torch.nn.Module`.
        """
        return self.model
