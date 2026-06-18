"""Job retrieval and lifecycle routes for the Argus Vision API.

Exposes endpoints to fetch a full :class:`~core.models.JobResult`, to query
just its status, and to delete a job (along with its temporary image file).
Missing jobs surface as :class:`~core.exceptions.JobNotFoundError`, which is
translated to an HTTP response by the application's global exception handler.
"""

import os

from fastapi import APIRouter, Request

from core.models import JobResult

router = APIRouter()

TEMP_IMAGE_DIR: str = "/tmp/argus"


@router.get("/jobs/{job_id}", response_model=JobResult)
async def get_job(request: Request, job_id: str) -> JobResult:
    """Return the complete result record for a job.

    Args:
        request: The incoming request, used to access the shared application
            singletons stored on ``request.app.state``.
        job_id: The identifier of the job to retrieve.

    Returns:
        The full :class:`~core.models.JobResult` for the requested job.

    Raises:
        JobNotFoundError: If no job exists for ``job_id`` (handled globally).
    """
    job_service = request.app.state.job_service
    job: JobResult = await job_service.get_job(job_id)
    return job


@router.get("/jobs/{job_id}/status")
async def get_job_status(request: Request, job_id: str) -> dict:
    """Return only the current status of a job.

    Args:
        request: The incoming request, used to access the shared application
            singletons stored on ``request.app.state``.
        job_id: The identifier of the job to query.

    Returns:
        A mapping with the single key ``status`` holding the job's current
        :class:`~core.models.JobStatus` value.

    Raises:
        JobNotFoundError: If no job exists for ``job_id`` (handled globally).
    """
    job_service = request.app.state.job_service
    job: JobResult = await job_service.get_job(job_id)
    return {"status": job.status}


@router.delete("/jobs/{job_id}")
async def delete_job(request: Request, job_id: str) -> dict:
    """Delete a job record and its temporary image file.

    Removes the job from the store and best-effort deletes the associated
    ``/tmp/argus/{job_id}.jpg`` file if it still exists on disk.

    Args:
        request: The incoming request, used to access the shared application
            singletons stored on ``request.app.state``.
        job_id: The identifier of the job to delete.

    Returns:
        A mapping with the single key ``deleted`` set to ``True``.

    Raises:
        JobNotFoundError: If no job exists for ``job_id`` (handled globally).
    """
    job_service = request.app.state.job_service
    await job_service.delete_job(job_id)

    image_path: str = os.path.join(TEMP_IMAGE_DIR, f"{job_id}.jpg")
    if os.path.exists(image_path):
        os.remove(image_path)

    return {"deleted": True}
