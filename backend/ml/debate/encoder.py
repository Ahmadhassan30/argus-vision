"""Sentence embedding of debate arguments for the consensus head.

The consensus MLP consumes a 788-dimensional feature vector that includes two
384-dimensional argument embeddings (``eA`` and ``eB``). This module wraps a
``sentence-transformers`` ``all-MiniLM-L6-v2`` model to produce those
L2-normalised 384-d embeddings.

The model is loaded lazily on first :meth:`ArgumentEncoder.encode` call so that
process startup stays fast, and any load or inference failure degrades to a
zero-vector of the correct length so the pipeline always produces a
well-formed feature vector.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Embedding dimensionality of the all-MiniLM-L6-v2 model.
EMBEDDING_DIM: int = 384

# Hugging Face model identifier for the sentence encoder.
MODEL_NAME: str = "all-MiniLM-L6-v2"


class ArgumentEncoder:
    """Encode debate argument text into fixed-length sentence embeddings.

    The underlying ``SentenceTransformer`` is instantiated lazily on the first
    :meth:`encode` call. It is placed on CUDA when a GPU is available and on CPU
    otherwise. If the model cannot be loaded or an encode call raises, the
    encoder returns a zero-vector of length :data:`EMBEDDING_DIM` and logs a
    warning, guaranteeing the consensus feature vector keeps its expected shape.

    Attributes:
        device: The torch device string the model is placed on
            (``"cuda"`` or ``"cpu"``).
    """

    def __init__(self) -> None:
        """Initialise the encoder without loading the model.

        The expensive model load is deferred until the first :meth:`encode`
        call to keep import and service-startup latency low.
        """
        self._model: Optional[object] = None
        self._load_failed: bool = False
        self.device: str = self._select_device()

    @staticmethod
    def _select_device() -> str:
        """Choose the best available torch device for inference.

        Returns:
            ``"cuda"`` when a CUDA device is available, otherwise ``"cpu"``.
            Falls back to ``"cpu"`` if torch cannot be imported.
        """
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001 - torch import/availability is best-effort.
            return "cpu"

    def _ensure_model(self) -> Optional[object]:
        """Lazily load the SentenceTransformer model on first use.

        Subsequent calls return the cached model. If a previous load attempt
        failed the method short-circuits and returns ``None`` so callers fall
        back to a zero-vector without repeatedly retrying an impossible load.

        Returns:
            The loaded ``SentenceTransformer`` instance, or ``None`` if loading
            has failed.
        """
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(MODEL_NAME, device=self.device)
            logger.info(
                "Loaded sentence encoder '%s' on device '%s'.",
                MODEL_NAME,
                self.device,
            )
            return self._model
        except Exception as exc:  # noqa: BLE001 - degrade gracefully on load error.
            self._load_failed = True
            logger.warning(
                "Failed to load sentence encoder '%s' (%s); "
                "argument embeddings will be zero-vectors.",
                MODEL_NAME,
                exc,
            )
            return None

    def encode(self, text: str) -> list[float]:
        """Encode a single argument string into a 384-d embedding.

        The embedding is L2-normalised (``normalize_embeddings=True``) so it can
        be concatenated directly into the consensus feature vector. Empty input
        or any failure yields a zero-vector of length :data:`EMBEDDING_DIM`.

        Args:
            text: The argument text to embed.

        Returns:
            A list of :data:`EMBEDDING_DIM` floats.
        """
        if not text or not text.strip():
            return [0.0] * EMBEDDING_DIM

        model = self._ensure_model()
        if model is None:
            return [0.0] * EMBEDDING_DIM

        try:
            embedding = model.encode(text, normalize_embeddings=True)
            vector = embedding.tolist()
            if len(vector) != EMBEDDING_DIM:
                logger.warning(
                    "Sentence encoder produced %d dims (expected %d); "
                    "returning zero-vector.",
                    len(vector),
                    EMBEDDING_DIM,
                )
                return [0.0] * EMBEDDING_DIM
            return [float(value) for value in vector]
        except Exception as exc:  # noqa: BLE001 - degrade gracefully on encode error.
            logger.warning(
                "Sentence encoding failed (%s); returning zero-vector.", exc
            )
            return [0.0] * EMBEDDING_DIM
