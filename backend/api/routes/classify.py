"""Classification entrypoint route for the Argus Vision API.

Exposes ``POST /classify`` which accepts an uploaded dermoscopic image,
persists it to a temporary location, registers a new job in the job store,
and schedules the adversarial debate pipeline to run asynchronously in the
background. The endpoint returns immediately with a job identifier that the
client can use to poll for results or to subscribe to the live debate stream.
"""

import asyncio
import os
import uuid

from fastapi import APIRouter, File, Request, UploadFile

from core.config import get_settings
from core.exceptions import ImageProcessingError, ModelNotLoadedError

router = APIRouter()

ALLOWED_CONTENT_TYPES: set[str] = {"image/jpeg", "image/png"}
TEMP_IMAGE_DIR: str = "/tmp/argus"


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
        ImageProcessingError: If the content type is unsupported or the image
            exceeds the maximum allowed size.
        ModelNotLoadedError: If the debate pipeline failed to initialise at
            startup and the service is running in a degraded state.
    """
    settings = get_settings()

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise ImageProcessingError(
            f"Unsupported content type '{file.content_type}'. "
            "Expected one of: image/jpeg, image/png."
        )

    contents: bytes = await file.read()

    max_bytes: int = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise ImageProcessingError(
            f"Image too large: {len(contents)} bytes exceeds the maximum of "
            f"{max_bytes} bytes ({settings.MAX_IMAGE_SIZE_MB} MB)."
        )

    job_id: str = str(uuid.uuid4())

    os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)
    image_path: str = os.path.join(TEMP_IMAGE_DIR, f"{job_id}.jpg")
    with open(image_path, "wb") as image_file:
        image_file.write(contents)

    job_service = request.app.state.job_service
    pipeline = request.app.state.pipeline

    if pipeline is None:
        raise ModelNotLoadedError(
            "Model pipeline is not available; the service started in a "
            "degraded state. Check the backend logs and model checkpoints."
        )

    await job_service.create_job(job_id, image_path)

    asyncio.create_task(pipeline.run(job_id, image_path, job_service))

    return {
        "job_id": job_id,
        "status": "queued",
        "estimated_seconds": 10,
    }
