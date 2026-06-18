"""Argus Vision API — FastAPI application entrypoint.

Adversarial multi-agent visual debate for uncertainty-aware medical image
classification. This module wires together the FastAPI application: lifespan
management for the ML pipeline and the Redis-backed job service, CORS,
structured request logging, global exception handling, and route registration.

Dependency injection note
--------------------------
To avoid circular imports between this module and the route/websocket modules,
the per-route dependency accessors (``get_pipeline`` / ``get_job_service``) are
defined *locally inside each route module* using ``starlette`` ``Request`` and
its ``request.app.state`` attribute (``request.app.state.pipeline`` and
``request.app.state.job_service``). The singletons are created here during the
lifespan startup and stored on ``app.state`` precisely so the route modules can
read them without importing ``main``. The accessors are also exported from this
module (``get_pipeline`` / ``get_job_service``) for convenience and for any
caller that prefers to import them from here.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import RequestResponseEndpoint

from api.routes import classify, health, jobs
from api.websocket import debate_stream
from core.config import get_settings
from core.exceptions import ArgusError, ModelNotLoadedError
from ml.pipeline import DebatePipeline
from services.job_service import JobService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("argus.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown lifecycle.

    On startup, instantiate the Redis-backed :class:`JobService` and the
    :class:`DebatePipeline` singleton, attaching both to ``app.state`` so that
    the route and websocket modules can resolve them via ``request.app.state``.
    Model-load failures are caught and logged so the service still boots; the
    health endpoint then reports ``model_loaded=false``.

    On shutdown, the Redis connection is closed and the pipeline is released.

    :param app: The FastAPI application instance whose ``state`` is populated.
    :yield: Control to the running application.
    """
    settings = get_settings()

    # Job service / Redis client. Constructing this should not load models.
    job_service = JobService()
    app.state.job_service = job_service

    # ML pipeline singleton. Model loading may fail (missing checkpoints, etc.);
    # we still want the API to come up so /health can report degraded status.
    pipeline: DebatePipeline | None = None
    try:
        pipeline = DebatePipeline()
        logger.info("DebatePipeline initialised (model_loaded=%s).", True)
    except Exception as exc:  # noqa: BLE001 — intentional broad catch on startup.
        logger.error(
            "Failed to initialise DebatePipeline (model_loaded=false): %s",
            exc,
            exc_info=True,
        )
        pipeline = None
    app.state.pipeline = pipeline

    logger.info(
        "Argus Vision API ready (redis_url=%s, model_loaded=%s).",
        settings.REDIS_URL,
        pipeline is not None,
    )

    try:
        yield
    finally:
        # Graceful shutdown: close Redis connection and release pipeline.
        try:
            await job_service.close()
            logger.info("JobService Redis connection closed.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Error closing JobService: %s", exc, exc_info=True)

        if pipeline is not None:
            try:
                close = getattr(pipeline, "close", None)
                if callable(close):
                    result = close()
                    if hasattr(result, "__await__"):
                        await result
                logger.info("DebatePipeline cleaned up.")
            except Exception as exc:  # noqa: BLE001
                logger.error("Error during pipeline cleanup: %s", exc, exc_info=True)

        logger.info("Argus Vision API shutdown complete.")


app = FastAPI(
    title="Argus Vision API",
    version="1.0.0",
    description=(
        "Adversarial multi-agent visual debate for uncertainty-aware medical "
        "image classification."
    ),
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next: RequestResponseEndpoint,
) -> JSONResponse:
    """Log every HTTP request with method, path, status code, and duration.

    :param request: The incoming HTTP request.
    :param call_next: The downstream ASGI handler that produces the response.
    :return: The response produced by the downstream handler.
    """
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "%s %s -> %s (%.2f ms)",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.exception_handler(ArgusError)
async def argus_error_handler(request: Request, exc: ArgusError) -> JSONResponse:
    """Translate domain :class:`ArgusError` instances into JSON responses.

    :param request: The request that triggered the error.
    :param exc: The raised :class:`ArgusError`.
    :return: JSON body ``{"error": <class name>, "detail": <detail>}`` with the
        error's ``status_code``.
    """
    logger.warning(
        "ArgusError on %s %s: %s (%s)",
        request.method,
        request.url.path,
        exc.detail,
        exc.status_code,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.__class__.__name__, "detail": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Return a structured 500 response for any unhandled exception.

    :param request: The request that triggered the error.
    :param exc: The unhandled exception.
    :return: JSON body ``{"error": "InternalServerError", "detail": str(exc)}``
        with HTTP status 500.
    """
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"error": "InternalServerError", "detail": str(exc)},
    )


def get_pipeline(request: Request) -> DebatePipeline:
    """Resolve the :class:`DebatePipeline` singleton from application state.

    Exposed as a convenience dependency accessor. Route modules typically define
    an identical helper locally to avoid importing this module, but importing
    from here is equally valid.

    :param request: The incoming request whose ``app.state`` holds the pipeline.
    :return: The active :class:`DebatePipeline`.
    :raises ArgusError: If the pipeline failed to load at startup.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise ModelNotLoadedError("Model pipeline is not available.")
    return pipeline


def get_job_service(request: Request) -> JobService:
    """Resolve the :class:`JobService` singleton from application state.

    :param request: The incoming request whose ``app.state`` holds the service.
    :return: The active :class:`JobService`.
    """
    return request.app.state.job_service


app.include_router(health.router)
app.include_router(classify.router)
app.include_router(jobs.router)
app.include_router(debate_stream.router)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
