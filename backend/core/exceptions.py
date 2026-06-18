"""Domain exceptions for the Argus Vision backend.

Every exception derives from :class:`ArgusError`, which carries an HTTP
``status_code`` class attribute and a human-readable ``detail`` message. FastAPI
exception handlers can translate any :class:`ArgusError` into an HTTP response
by reading these two attributes uniformly.
"""

from __future__ import annotations


class ArgusError(Exception):
    """Base class for all Argus Vision domain errors.

    Attributes:
        status_code: HTTP status code associated with the error. Subclasses
            override this with a semantically appropriate code.
        detail: Human-readable description of what went wrong. Defaults to a
            generic message and can be overridden per-instance.
    """

    status_code: int = 500
    default_detail: str = "An unexpected error occurred."

    def __init__(self, detail: str | None = None) -> None:
        """Initialise the error with an optional override detail message.

        Args:
            detail: Explanation of the failure. When omitted, the subclass'
                :attr:`default_detail` is used.
        """
        self.detail: str = detail if detail is not None else self.default_detail
        super().__init__(self.detail)


class ModelNotLoadedError(ArgusError):
    """Raised when an ML model is required but has not been loaded.

    Maps to HTTP 503 Service Unavailable because the service cannot fulfil the
    request until its models are available.
    """

    status_code: int = 503
    default_detail: str = "A required model has not been loaded."


class JobNotFoundError(ArgusError):
    """Raised when a job identifier does not correspond to a stored job.

    Maps to HTTP 404 Not Found.
    """

    status_code: int = 404
    default_detail: str = "The requested job was not found."


class ImageProcessingError(ArgusError):
    """Raised when an uploaded image cannot be decoded or processed.

    Maps to HTTP 422 Unprocessable Entity because the input was syntactically
    received but semantically invalid.
    """

    status_code: int = 422
    default_detail: str = "The image could not be processed."


class DebateError(ArgusError):
    """Raised when the multi-agent debate pipeline fails.

    Maps to HTTP 502 Bad Gateway, reflecting that an upstream dependency (such
    as the LLM provider) misbehaved during debate generation.
    """

    status_code: int = 502
    default_detail: str = "The debate process failed."
