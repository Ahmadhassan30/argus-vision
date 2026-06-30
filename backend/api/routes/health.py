"""Health check route for the Argus Vision API.

Exposes a single ``GET /health`` endpoint that reports the liveness of the
service together with the readiness of its critical dependencies (the ML
pipeline and the Redis backing store).
"""

from datetime import datetime

from fastapi import APIRouter, Request

from core.config import get_settings

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict:
    """Report service health and dependency readiness.

    Checks whether the ML debate pipeline has loaded its model weights and
    whether the Redis-backed job store is reachable. Neither check raises:
    a degraded dependency is reported as ``False`` rather than failing the
    endpoint, so orchestrators can still scrape the probe.

    Args:
        request: The incoming request, used to access the shared application
            singletons stored on ``request.app.state``.

    Returns:
        A mapping containing the overall ``status``, the API ``version``, the
        ``model_loaded`` flag, the ``redis_connected`` flag, and an ISO-8601
        UTC ``timestamp``.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    model_loaded: bool = pipeline is not None and pipeline.is_loaded()

    redis_connected: bool = False
    job_service = getattr(request.app.state, "job_service", None)
    if job_service is not None:
        try:
            redis_connected = await job_service.ping()
        except Exception:
            redis_connected = False

    return {
        "status": "ok",
        "version": get_settings().APP_VERSION,
        "model_loaded": model_loaded,
        "redis_connected": redis_connected,
        "timestamp": datetime.utcnow().isoformat(),
    }
