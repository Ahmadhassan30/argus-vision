"""Consensus classifier MLP for the Argus Vision adversarial debate.

After the two agents have classified the lesion and (optionally) debated, the
consensus head fuses every available signal into a single calibrated prediction.
It consumes a fixed 788-dimensional feature vector formed by concatenating, in
this exact contract order:

* ``pA`` — Agent A's 8-class probability distribution.
* ``pB`` — Agent B's 8-class probability distribution.
* ``spatial_stats`` — ``[mean_a, mean_b, std_a, std_b]`` taken from the contested
  attention region (zeros on the non-debate fast path).
* ``eA`` — Agent A's 384-d argument sentence embedding (zeros when no debate).
* ``eB`` — Agent B's 384-d argument sentence embedding (zeros when no debate).

The MLP (``788 -> 512 -> 256 -> 8``) emits logits that are divided by a learnable
temperature scalar before the softmax, giving a calibrated distribution. The
calibration error (ECE) is computed at training time and exposed here as a
settable attribute that flows into the produced :class:`ConsensusResult`.
"""

from __future__ import annotations

import logging
import os
from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models import ConsensusResult

logger = logging.getLogger(__name__)

# ISIC-8 class names in their canonical (index 0..7) order.
CLASS_NAMES: list[str] = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# Number of output classes for the consensus head.
NUM_CLASSES: int = 8

# Dimensionality of the concatenated consensus feature vector.
FEATURE_DIM: int = 788

# Component dimensionalities, in concatenation order: pA + pB + spatial + eA + eB.
PROB_DIM: int = 8
SPATIAL_DIM: int = 4
EMBED_DIM: int = 384

# Lower bound applied to the temperature before it divides the logits, so a
# degenerate (zero/negative) learned temperature can never blow up the softmax.
_TEMPERATURE_FLOOR: float = 1e-2

# Type alias for the per-component inputs accepted by :meth:`forward`.
VectorInput = Union[torch.Tensor, Sequence[float]]


class ConsensusClassifier(nn.Module):
    """Calibrated MLP that fuses agent probabilities, spatial and text signals.

    The network architecture is fixed by the shared contract::

        Linear(788, 512) -> BatchNorm1d(512) -> ReLU -> Dropout(0.3)
        Linear(512, 256) -> BatchNorm1d(256) -> ReLU -> Dropout(0.3)
        Linear(256, 8)

    A single learnable temperature parameter (initialised to ``1.0``) divides the
    output logits prior to the softmax for post-hoc calibration.

    Attributes:
        device: The resolved torch device string (``"cuda"`` or ``"cpu"``).
        mlp: The sequential feature-fusion network producing raw logits.
        temperature: A learnable scalar (``nn.Parameter``) used for calibration.
        calibration_ece: Expected calibration error measured at train time; this
            value is surfaced in every :class:`ConsensusResult` produced by
            :meth:`predict`. Defaults to ``0.0`` for an uncalibrated model.
    """

    def __init__(
        self,
        checkpoint_path: str | None,
        device: str | None = None,
    ) -> None:
        """Build the consensus MLP and optionally load trained weights.

        Args:
            checkpoint_path: Path to a trained ``.pth`` state dict. When the file
                exists it is loaded with ``strict=False`` and the load result is
                logged. When it does not exist the network is left with its
                randomly-initialised weights and a warning is emitted noting that
                consensus predictions are unreliable until the head is trained.
            device: Optional torch device string. When ``None`` the device is
                resolved automatically to ``"cuda"`` if available else ``"cpu"``.
        """
        super().__init__()

        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Expected calibration error measured at training time; settable so a
        # training script can stamp the value onto the saved/loaded model.
        self.calibration_ece: float = 0.0

        # 788 -> 512 -> 256 -> 8 with BatchNorm1d + ReLU + Dropout(0.3) after
        # each of the two hidden layers, exactly as specified by the contract.
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(FEATURE_DIM, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES),
        )

        # Learnable temperature scalar (init 1.0) for post-hoc calibration.
        self.temperature: nn.Parameter = nn.Parameter(torch.ones(1))

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            logger.info(
                "ConsensusClassifier: loading checkpoint from '%s' onto device "
                "'%s'.",
                checkpoint_path,
                self.device,
            )
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                # Allow the optional calibration metric to ride along inside the
                # checkpoint envelope (e.g. {"state_dict": ..., "ece": ...}).
                if "ece" in state_dict:
                    try:
                        self.calibration_ece = float(state_dict["ece"])
                    except (TypeError, ValueError):
                        self.calibration_ece = 0.0
                state_dict = state_dict["state_dict"]
            load_result = self.load_state_dict(state_dict, strict=False)
            if load_result.missing_keys:
                logger.warning(
                    "ConsensusClassifier: %d missing key(s) when loading "
                    "checkpoint: %s",
                    len(load_result.missing_keys),
                    load_result.missing_keys,
                )
            if load_result.unexpected_keys:
                logger.warning(
                    "ConsensusClassifier: %d unexpected key(s) when loading "
                    "checkpoint: %s",
                    len(load_result.unexpected_keys),
                    load_result.unexpected_keys,
                )
        else:
            logger.warning(
                "ConsensusClassifier: no checkpoint found at '%s'. The fusion "
                "head is randomly initialised and UNTRAINED; consensus "
                "predictions are unreliable until the head has been trained.",
                checkpoint_path,
            )

        self.eval()
        self.to(self.device)

    def _to_tensor(self, values: VectorInput, expected_dim: int, name: str) -> torch.Tensor:
        """Coerce a per-component input into a 1-D float tensor on the device.

        Args:
            values: Either a 1-D ``torch.Tensor`` (any leading batch dimension is
                flattened away) or a sequence of floats.
            expected_dim: The number of elements the component must contain.
            name: Human-readable component name, used only in error messages.

        Returns:
            A ``float32`` 1-D tensor of length ``expected_dim`` on ``self.device``.

        Raises:
            ValueError: If the flattened input does not contain exactly
                ``expected_dim`` elements.
        """
        if isinstance(values, torch.Tensor):
            tensor = values.detach().to(dtype=torch.float32).reshape(-1)
        else:
            tensor = torch.tensor(list(values), dtype=torch.float32)

        if tensor.numel() != expected_dim:
            raise ValueError(
                f"Consensus component '{name}' must have {expected_dim} elements, "
                f"got {tensor.numel()}."
            )
        return tensor.to(self.device)

    def _build_feature_vector(
        self,
        pa: VectorInput,
        pb: VectorInput,
        spatial_stats: VectorInput,
        ea: VectorInput,
        eb: VectorInput,
    ) -> torch.Tensor:
        """Assemble the 788-d consensus feature vector in contract order.

        Args:
            pa: Agent A's 8-class probability distribution.
            pb: Agent B's 8-class probability distribution.
            spatial_stats: ``[mean_a, mean_b, std_a, std_b]`` (4 values).
            ea: Agent A's 384-d argument embedding.
            eb: Agent B's 384-d argument embedding.

        Returns:
            A ``float32`` tensor of shape ``(1, 788)`` on ``self.device``.
        """
        components = [
            self._to_tensor(pa, PROB_DIM, "pA"),
            self._to_tensor(pb, PROB_DIM, "pB"),
            self._to_tensor(spatial_stats, SPATIAL_DIM, "spatial_stats"),
            self._to_tensor(ea, EMBED_DIM, "eA"),
            self._to_tensor(eb, EMBED_DIM, "eB"),
        ]
        feature = torch.cat(components, dim=0)
        return feature.unsqueeze(0)

    def forward(
        self,
        pa: VectorInput,
        pb: VectorInput,
        spatial_stats: VectorInput,
        ea: VectorInput,
        eb: VectorInput,
    ) -> torch.Tensor:
        """Run the fusion MLP and return temperature-scaled probabilities.

        The five components are concatenated in the contract order
        (``pA + pB + spatial_stats + eA + eB``) into a 788-vector, passed through
        the MLP to obtain logits, divided by the clamped temperature, and softmaxed.

        Args:
            pa: Agent A's 8-class probability distribution (tensor or sequence).
            pb: Agent B's 8-class probability distribution (tensor or sequence).
            spatial_stats: ``[mean_a, mean_b, std_a, std_b]`` (tensor or sequence).
            ea: Agent A's 384-d argument embedding (tensor or sequence).
            eb: Agent B's 384-d argument embedding (tensor or sequence).

        Returns:
            A probability tensor. The batch dimension is squeezed away when the
            inputs describe a single sample, yielding shape ``[8]``; batched
            inputs would yield ``[B, 8]``.
        """
        features = self._build_feature_vector(pa, pb, spatial_stats, ea, eb)
        logits = self.mlp(features)

        # Clamp the temperature so a degenerate learned value cannot destabilise
        # the softmax, then scale the logits before normalising.
        temperature = torch.clamp(self.temperature, min=_TEMPERATURE_FLOOR)
        scaled_logits = logits / temperature

        probabilities = F.softmax(scaled_logits, dim=1)

        # Single-sample inputs collapse the batch dimension to a flat [8] vector.
        if probabilities.shape[0] == 1:
            return probabilities.squeeze(0)
        return probabilities

    def predict(
        self,
        pa: VectorInput,
        pb: VectorInput,
        spatial_stats: VectorInput,
        ea: VectorInput,
        eb: VectorInput,
    ) -> ConsensusResult:
        """Produce a structured, calibrated consensus prediction.

        Runs :meth:`forward` under ``torch.no_grad()``, then maps the resulting
        distribution onto the ISIC-8 labels and reports the argmax class, its
        probability, the learned temperature, and the model's calibration error.

        Args:
            pa: Agent A's 8-class probability distribution.
            pb: Agent B's 8-class probability distribution.
            spatial_stats: ``[mean_a, mean_b, std_a, std_b]`` from the contested
                region (or zeros on the non-debate fast path).
            ea: Agent A's 384-d argument embedding (or a zero-vector).
            eb: Agent B's 384-d argument embedding (or a zero-vector).

        Returns:
            A :class:`core.models.ConsensusResult` with the predicted class, its
            confidence, the full probability mapping, the learned temperature and
            the configured calibration error (:attr:`calibration_ece`).
        """
        with torch.no_grad():
            probabilities = self.forward(pa, pb, spatial_stats, ea, eb)

        prob_row = probabilities.reshape(-1)
        top_index = int(torch.argmax(prob_row).item())

        probability_map: dict[str, float] = {
            class_name: float(prob_row[idx].item())
            for idx, class_name in enumerate(CLASS_NAMES)
        }

        return ConsensusResult(
            pred_class=CLASS_NAMES[top_index],
            confidence=float(prob_row[top_index].item()),
            probabilities=probability_map,
            temperature=float(self.temperature.detach().reshape(-1)[0].item()),
            ece=float(self.calibration_ece),
        )
