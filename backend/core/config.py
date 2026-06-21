"""Application configuration for the Argus Vision backend.

This module defines the :class:`Settings` object backed by
``pydantic-settings``. Every field is sourced from an environment variable
whose name is the uppercased field name (the names already match because the
fields are declared in uppercase). A cached :func:`get_settings` accessor is
provided so the settings object is constructed exactly once per process.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the Argus Vision service.

    Values are read from the process environment (and, as a fallback, from a
    ``.env`` file in the working directory). Unknown environment variables are
    ignored so the container can carry additional, unrelated configuration
    without breaking startup.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- External services --------------------------------------------------
    REDIS_URL: str = "redis://redis:6379"
    """Connection URL for the Redis instance backing jobs and pub/sub."""

    # --- Model checkpoints --------------------------------------------------
    MODEL_CHECKPOINT_DIR: str = "./checkpoints"
    """Directory containing the trained model checkpoint files."""

    AGENT_A_CHECKPOINT: str = "agent_a_best.pth"
    """Filename of the Agent A (EfficientNet-B4) checkpoint."""

    AGENT_B_CHECKPOINT: str = "agent_b_best.pth"
    """Filename of the Agent B (ViT-B/16) checkpoint."""

    CONSENSUS_CHECKPOINT: str = "consensus_best.pth"
    """Filename of the consensus MLP checkpoint."""

    CONSENSUS_SCALER: str = "consensus_scaler.pkl"
    """Filename of the StandardScaler fitted on the consensus training split.

    A ``consensus_scaler.json`` ``{"mean", "scale"}`` sidecar next to it is used
    as a numpy-only fallback when scikit-learn/joblib is unavailable."""

    PRETRAINED_FALLBACK: bool = True
    """If true, fall back to ImageNet-pretrained weights when a checkpoint
    is missing instead of raising an error."""

    # --- Debate trigger thresholds -----------------------------------------
    DEBATE_JS_THRESHOLD: float = 0.25
    """Jensen-Shannon divergence above which a debate is triggered."""

    DEBATE_ENTROPY_THRESHOLD: float = 0.8
    """Prediction entropy above which spatial attention + consensus run."""

    # --- Upload / request limits -------------------------------------------
    MAX_IMAGE_SIZE_MB: int = 10
    """Maximum accepted upload size for an image, in megabytes."""

    # --- CORS ---------------------------------------------------------------
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost"
    """Comma-separated list of origins permitted by CORS."""

    @property
    def allowed_origins_list(self) -> list[str]:
        """Return :attr:`ALLOWED_ORIGINS` as a cleaned list of origins.

        The raw string is split on commas, surrounding whitespace is stripped
        from each entry, and empty entries are discarded.

        Returns:
            A list of non-empty, whitespace-trimmed origin strings.
        """
        return [
            origin.strip()
            for origin in self.ALLOWED_ORIGINS.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings` instance.

    The result is memoised with :func:`functools.lru_cache` so environment
    parsing happens only once and every caller shares the same object.

    Returns:
        The singleton :class:`Settings` instance.
    """
    return Settings()
