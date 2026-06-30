"""Integration tests for ``POST /classify`` via FastAPI's TestClient.

The route reads its singletons straight off ``request.app.state`` (no FastAPI
``Depends``), so these tests inject an ``AsyncMock`` job service and a
``MagicMock`` pipeline with an async ``.run`` directly onto ``main.app.state``.
The client is intentionally *not* used as a context manager so the real lifespan
(Redis + model loading) never runs.

Pinned contract:
  * happy path -> 200 with exactly {"job_id", "status", "estimated_seconds"},
    a valid UUID job id, status "queued", and an int estimate.
  * unsupported content type -> 422, {"error": "ImageProcessingError", ...}.
  * corrupt-but-typed bytes -> 422 (the one sanctioned Phase-9 contract change:
    a synchronous Pillow verify), same error envelope.

Requires fastapi + httpx + Pillow installed to run (not available in the audit
env). The runtime deps come from requirements.txt; test-only deps from
requirements-dev.txt.
"""

from __future__ import annotations

import io
import os
import sys
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Make ``main`` importable regardless of pytest's invocation dir.
_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import main  # noqa: E402


def _png_bytes() -> bytes:
    """Return the bytes of a valid tiny RGB PNG that Pillow can verify."""
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (123, 222, 64)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client() -> TestClient:
    """A TestClient with mocked app.state singletons (no real lifespan)."""
    main.app.state.job_service = AsyncMock()
    pipeline = MagicMock()
    pipeline.run = AsyncMock()
    main.app.state.pipeline = pipeline
    # Not a context manager: lifespan (Redis/model load) is deliberately skipped.
    return TestClient(main.app)


def test_valid_png_returns_queued_job(client: TestClient) -> None:
    """A valid PNG upload is accepted and queued."""
    response = client.post(
        "/classify",
        files={"file": ("lesion.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    # Response shape is part of the external contract — lock the exact key set.
    assert set(body) == {"job_id", "status", "estimated_seconds"}
    uuid.UUID(body["job_id"])  # raises if not a valid UUID
    assert body["status"] == "queued"
    assert isinstance(body["estimated_seconds"], int)


def test_unsupported_content_type_returns_422(client: TestClient) -> None:
    """A non-image content type is rejected with the ImageProcessingError envelope."""
    response = client.post(
        "/classify",
        files={"file": ("note.txt", b"hello world", "text/plain")},
    )

    assert response.status_code == 422
    assert response.json()["error"] == "ImageProcessingError"


def test_corrupt_image_returns_422(client: TestClient) -> None:
    """Corrupt bytes with an image content type are rejected (Phase-9 verify).

    This is the one sanctioned contract change: previously the bytes were
    written and the pipeline scheduled (a 200); a synchronous Pillow verify now
    surfaces the bad image up-front as a 422 with the same error envelope.
    """
    response = client.post(
        "/classify",
        files={"file": ("bad.png", b"not an image", "image/png")},
    )

    assert response.status_code == 422
    assert response.json()["error"] == "ImageProcessingError"


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-q"]))
