"""Classification entrypoint route for the Argus Vision API.

Exposes ``POST /classify`` which accepts an uploaded dermoscopic image,
persists it to a temporary location, registers a new job in the job store,
and schedules the adversarial debate pipeline to run asynchronously in the
background. The endpoint returns immediately with a job identifier that the
client can use to poll for results or to subscribe to the live debate stream.
"""

import asyncio
import io
import logging
import os
import uuid

from fastapi import APIRouter, File, Request, UploadFile
from PIL import Image

from core.config import get_settings
from core.exceptions import ImageProcessingError, ModelNotLoadedError

router = APIRouter()
logger = logging.getLogger(__name__)

ALLOWED_CONTENT_TYPES: set[str] = {"image/jpeg", "image/png"}


@router.post("/classify")
async def classify(request: Request, file: UploadFile = File(...)) -> dict:
    """Accept an image upload and launch the debate pipeline.

    Validates the upload's content type and size, writes the bytes to
    ``/tmp/argus/{job_id}.jpg``, creates the job record, and schedules the
    pipeline as a fire-and-forget background task. The pipeline is *not*
    awaited so the request returns promptly.

    Args:
        request: The incoming request, used to access the shared application
            singletons stored on ``request.app.state``.
        file: The multipart-uploaded image (form field name ``file``). Must be
            a JPEG or PNG no larger than the configured maximum size.

    Returns:
        A mapping with the generated ``job_id``, a ``status`` of ``"queued"``,
        and an ``estimated_seconds`` hint of ``10``.

    Raises:
        ImageProcessingError: If the content type is unsupported, the image
            exceeds the maximum allowed size, or the bytes are not a valid
            decodable image.
        ModelNotLoadedError: If the debate pipeline failed to initialise at
            startup and the service is running in a degraded state.
    """
    settings = get_settings()

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise ImageProcessingError(
            f"Unsupported content type '{file.content_type}'. "
            "Expected one of: image/jpeg, image/png."
        )

    max_bytes: int = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024

    # Reject oversized uploads *before* buffering the whole body into memory.
    # Starlette populates ``UploadFile.size`` from the multipart part, so we can
    # bail out without reading gigabytes of attacker-controlled bytes.
    if file.size is not None and file.size > max_bytes:
        raise ImageProcessingError(
            f"Image too large: {file.size} bytes exceeds the maximum of "
            f"{max_bytes} bytes ({settings.MAX_IMAGE_SIZE_MB} MB)."
        )

    # Read at most one byte past the limit; getting that extra byte proves the
    # upload is over budget even when the size hint is absent or spoofed.
    contents: bytes = await file.read(max_bytes + 1)
    if len(contents) > max_bytes:
        raise ImageProcessingError(
            f"Image too large: {len(contents)} bytes exceeds the maximum of "
            f"{max_bytes} bytes ({settings.MAX_IMAGE_SIZE_MB} MB)."
        )

    # Verify the bytes decode as a real image so corrupt-but-typed uploads fail
    # synchronously here instead of crashing the pipeline later.
    try:
        Image.open(io.BytesIO(contents)).verify()
    except Exception as exc:
        raise ImageProcessingError("Uploaded file is not a valid image.") from exc

    job_id: str = str(uuid.uuid4())

    job_service = request.app.state.job_service
    pipeline = request.app.state.pipeline

    # Fail fast on a degraded service *before* writing a temp file; otherwise a
    # 503 would orphan an image on disk for every rejected request.
    if pipeline is None:
        raise ModelNotLoadedError(
            "Model pipeline is not available; the service started in a "
            "degraded state. Check the backend logs and model checkpoints."
        )

    os.makedirs(settings.TEMP_IMAGE_DIR, exist_ok=True)
    image_path: str = os.path.join(settings.TEMP_IMAGE_DIR, f"{job_id}.jpg")
    with open(image_path, "wb") as image_file:
        image_file.write(contents)

    try:
        await job_service.create_job(job_id, image_path)
    except Exception:
        # Don't leave the temp file behind if registering the job fails.
        if os.path.exists(image_path):
            os.remove(image_path)
        raise

    asyncio.create_task(pipeline.run(job_id, image_path, job_service))
    logger.info(
        "Scheduled debate pipeline for job %s (%d bytes)", job_id, len(contents)
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "estimated_seconds": settings.ESTIMATED_SECONDS,
    }
