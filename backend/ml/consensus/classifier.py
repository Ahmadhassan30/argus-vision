"""Consensus classifier MLP for the Argus Vision pipeline (23-dim contract).

After the two agents have classified the lesion, the consensus head fuses their
signals into a single calibrated prediction. It consumes the fixed
**23-dimensional** numerical feature vector built by
:func:`ml.debate.features.extract_consensus_features` (Agent A/B softmax
probabilities, their disagreement statistics, and their spatial-attention
agreement). The previous 788-dim Groq-debate-text + sentence-embedding contract
has been removed entirely.

The features are standardized with the same ``StandardScaler`` that was fit on
the consensus training split (saved as ``consensus_scaler.pkl`` next to the
checkpoint, with a ``consensus_scaler.json`` ``{"mean", "scale"}`` sidecar so the
serving path needs only numpy). The MLP (``23 -> 128 -> 64 -> 8``) emits logits
that are divided by a learnable temperature scalar before the softmax.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models import ConsensusResult
from ml.debate.features import FEATURE_DIM, extract_consensus_features

logger = logging.getLogger(__name__)

# ISIC-8 class names in their canonical (index 0..7) order.
CLASS_NAMES: list[str] = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

NUM_CLASSES: int = 8

# Lower bound applied to the temperature before it divides the logits.
_TEMPERATURE_FLOOR: float = 1e-2

VectorInput = Union[torch.Tensor, Sequence[float], np.ndarray]


class ConsensusClassifier(nn.Module):
    """Calibrated MLP that fuses the 23-dim consensus feature vector.

    Architecture (parameter names match the training checkpoint exactly so it
    loads with ``strict=True``-equivalent zero missing/unexpected keys)::

        Linear(23, 128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.3)
        Linear(128, 64) -> BatchNorm1d(64)  -> ReLU -> Dropout(0.3)
        Linear(64, 8)

    A single learnable temperature scalar divides the logits before the softmax.

    Attributes:
        device: Resolved torch device string.
        mlp: The fusion network producing raw logits.
        temperature: Learnable calibration scalar.
        calibration_ece: ECE measured at train time, surfaced in every result.
    """

    def __init__(
        self,
        checkpoint_path: str | None,
        scaler_path: str | None = None,
        device: str | None = None,
    ) -> None:
        """Build the consensus MLP, load weights, and load the feature scaler.

        Args:
            checkpoint_path: Path to a trained ``.pth`` state dict (or an
                envelope ``{"state_dict": ..., "ece": ...}``). When missing the
                head is left randomly initialised and a warning is emitted.
            scaler_path: Path to ``consensus_scaler.pkl`` (joblib) saved during
                training. A ``consensus_scaler.json`` sidecar next to it is used
                as a numpy-only fallback. When neither is found, an identity
                scaler is used and a warning is emitted.
            device: Optional torch device string; auto-resolved when ``None``.
        """
        super().__init__()

        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.calibration_ece: float = 0.0

        # 23 -> 128 -> 64 -> 8, matching the training notebook's ConsensusClassifier.
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(FEATURE_DIM, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, NUM_CLASSES),
        )
        self.temperature: nn.Parameter = nn.Parameter(torch.ones(1))

        # Feature standardization (mean / scale), defaulting to identity.
        self._scaler_mean: np.ndarray = np.zeros(FEATURE_DIM, dtype=np.float64)
        self._scaler_scale: np.ndarray = np.ones(FEATURE_DIM, dtype=np.float64)
        self._scaler_loaded: bool = False

        self._load_checkpoint(checkpoint_path)
        self._load_scaler(scaler_path, checkpoint_path)

        self.eval()
        self.to(self.device)

    # ------------------------------------------------------------------ loading
    def _load_checkpoint(self, checkpoint_path: str | None) -> None:
        """Load the MLP/temperature weights, tolerating the envelope format."""
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            logger.info(
                "ConsensusClassifier: loading checkpoint '%s' onto '%s'.",
                checkpoint_path,
                self.device,
            )
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                if "ece" in state_dict:
                    try:
                        self.calibration_ece = float(state_dict["ece"])
                    except (TypeError, ValueError):
                        self.calibration_ece = 0.0
                state_dict = state_dict["state_dict"]
            load_result = self.load_state_dict(state_dict, strict=False)
            if load_result.missing_keys:
                logger.warning(
                    "ConsensusClassifier: %d missing key(s): %s",
                    len(load_result.missing_keys),
                    load_result.missing_keys,
                )
            if load_result.unexpected_keys:
                logger.warning(
                    "ConsensusClassifier: %d unexpected key(s): %s",
                    len(load_result.unexpected_keys),
                    load_result.unexpected_keys,
                )
        else:
            logger.warning(
                "ConsensusClassifier: no checkpoint at '%s'. The fusion head is "
                "randomly initialised and UNTRAINED; predictions are unreliable.",
                checkpoint_path,
            )

    def _load_scaler(self, scaler_path: str | None, checkpoint_path: str | None) -> None:
        """Load the StandardScaler mean/scale (joblib .pkl, else .json sidecar).

        Args:
            scaler_path: Explicit path to ``consensus_scaler.pkl``.
            checkpoint_path: Used to derive a sibling scaler path when
                ``scaler_path`` is not supplied.
        """
        candidates: list[str] = []
        if scaler_path:
            candidates.append(scaler_path)
            candidates.append(os.path.splitext(scaler_path)[0] + ".json")
        if checkpoint_path:
            ckpt_dir = os.path.dirname(checkpoint_path)
            candidates.append(os.path.join(ckpt_dir, "consensus_scaler.pkl"))
            candidates.append(os.path.join(ckpt_dir, "consensus_scaler.json"))

        # Try the joblib .pkl form first (a real sklearn StandardScaler).
        for path in candidates:
            if path.endswith(".pkl") and os.path.exists(path):
                try:
                    import joblib

                    scaler = joblib.load(path)
                    mean = np.asarray(getattr(scaler, "mean_"), dtype=np.float64)
                    scale = np.asarray(getattr(scaler, "scale_"), dtype=np.float64)
                    if mean.shape == (FEATURE_DIM,) and scale.shape == (FEATURE_DIM,):
                        self._scaler_mean = mean
                        self._scaler_scale = np.where(scale == 0.0, 1.0, scale)
                        self._scaler_loaded = True
                        logger.info("ConsensusClassifier: loaded scaler from '%s'.", path)
                        return
                except Exception as exc:  # noqa: BLE001 - fall back to json/identity.
                    logger.warning(
                        "ConsensusClassifier: could not load joblib scaler '%s' "
                        "(%s); trying JSON sidecar.",
                        path,
                        exc,
                    )

        # Fall back to the numpy-only json sidecar.
        for path in candidates:
            if path.endswith(".json") and os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        payload = json.load(fh)
                    mean = np.asarray(payload["mean"], dtype=np.float64)
                    scale = np.asarray(payload["scale"], dtype=np.float64)
                    if mean.shape == (FEATURE_DIM,) and scale.shape == (FEATURE_DIM,):
                        self._scaler_mean = mean
                        self._scaler_scale = np.where(scale == 0.0, 1.0, scale)
                        self._scaler_loaded = True
                        logger.info("ConsensusClassifier: loaded scaler from '%s'.", path)
                        return
                except Exception as exc:  # noqa: BLE001 - fall back to identity.
                    logger.warning(
                        "ConsensusClassifier: could not load JSON scaler '%s' (%s).",
                        path,
                        exc,
                    )

        logger.warning(
            "ConsensusClassifier: no feature scaler found (looked for %s). Using "
            "an IDENTITY scaler; predictions will be miscalibrated because the MLP "
            "was trained on standardized features.",
            candidates,
        )

    # ------------------------------------------------------------------ inference
    def _standardize(self, feature: np.ndarray) -> torch.Tensor:
        """Apply the saved StandardScaler to a raw 23-d feature vector."""
        scaled = (feature.astype(np.float64) - self._scaler_mean) / self._scaler_scale
        return torch.tensor(scaled, dtype=torch.float32, device=self.device).unsqueeze(0)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """Run the MLP on an already-standardized ``(1, 23)`` feature tensor.

        Returns the temperature-scaled probability vector (squeezed to ``[8]``
        for a single sample).
        """
        logits = self.mlp(feature)
        temperature = torch.clamp(self.temperature, min=_TEMPERATURE_FLOOR)
        probabilities = F.softmax(logits / temperature, dim=1)
        if probabilities.shape[0] == 1:
            return probabilities.squeeze(0)
        return probabilities

    def predict(
        self,
        prob_a: VectorInput,
        prob_b: VectorInput,
        attn_map_a: Optional[np.ndarray] = None,
        attn_map_b: Optional[np.ndarray] = None,
    ) -> ConsensusResult:
        """Produce a calibrated consensus prediction from the agent signals.

        Builds the 23-d feature vector with the canonical extractor, standardizes
        it with the saved scaler, and runs the calibrated MLP.

        Args:
            prob_a: Agent A's 8-class probability distribution (canonical order).
            prob_b: Agent B's 8-class probability distribution (canonical order).
            attn_map_a: Agent A's 2D attention map, or ``None`` on the fast path.
            attn_map_b: Agent B's 2D attention map, or ``None`` on the fast path.

        Returns:
            A :class:`core.models.ConsensusResult` with the predicted class, its
            calibrated confidence, the full probability mapping, the learned
            temperature and the configured calibration error.
        """
        pa = np.asarray(list(prob_a) if not isinstance(prob_a, np.ndarray) else prob_a,
                        dtype=np.float64).ravel()
        pb = np.asarray(list(prob_b) if not isinstance(prob_b, np.ndarray) else prob_b,
                        dtype=np.float64).ravel()

        feature = extract_consensus_features(pa, pb, attn_map_a, attn_map_b)
        feature_tensor = self._standardize(feature)

        with torch.no_grad():
            probabilities = self.forward(feature_tensor)

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
