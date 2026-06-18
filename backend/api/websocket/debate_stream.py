"""WebSocket endpoint that streams Argus Vision debate events to clients.

This module defines a single WebSocket route, ``/ws/debate/{job_id}``, which:

1. Accepts the socket and verifies that the referenced job exists by querying
   the :class:`~services.job_service.JobService` stored on the application
   state. Unknown jobs cause the socket to be closed with code ``4004``.
2. Immediately replays the current :class:`~core.models.JobResult` snapshot so
   that late subscribers can catch up on any progress that has already been
   computed before they connected.
3. Subscribes to the Redis pub/sub channel ``argus:debate:{job_id}`` and relays
   every published message (already serialized JSON text) verbatim to the
   client via ``send_text``.
4. Runs a concurrent keepalive task that emits a ``{"type": "ping"}`` frame
   every 30 seconds to keep idle connections alive through proxies.
5. Cleans up the pub/sub subscription on disconnect in a ``finally`` block.

The published payloads conform to the ``DebateEvent`` discriminated union in
``core.models`` and are forwarded without re-serialization.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.models import JobResult
from services.job_service import JobNotFoundError, JobService

if TYPE_CHECKING:  # pragma: no cover - imported for typing only
    from redis import asyncio as aioredis

logger = logging.getLogger("argus.websocket.debate")

router = APIRouter()

# Interval, in seconds, between keepalive ping frames sent to the client.
KEEPALIVE_INTERVAL_SECONDS: float = 30.0

# WebSocket close code used when the requested job cannot be found.
WS_CLOSE_JOB_NOT_FOUND: int = 4004


async def _send_snapshot(websocket: WebSocket, job: JobResult) -> None:
    """Send the current job snapshot to the freshly connected client.

    Serializing the snapshot first allows clients that connect mid-pipeline to
    render whatever state has already been computed before they begin receiving
    live pub/sub events.

    Args:
        websocket: The accepted client WebSocket connection.
        job: The current :class:`~core.models.JobResult` for the subscribed job.
    """
    await websocket.send_text(job.model_dump_json())


async def _relay_messages(
    websocket: WebSocket,
    pubsub: "aioredis.client.PubSub",
) -> None:
    """Relay Redis pub/sub messages to the WebSocket client.

    Listens on the already-subscribed pub/sub object and forwards the payload of
    every ``message``-type event to the client unchanged. Payloads published to
    the channel are JSON-encoded :class:`~core.models.DebateEvent` strings, so
    they are sent verbatim via ``send_text``.

    Args:
        websocket: The accepted client WebSocket connection.
        pubsub: A Redis pub/sub object already subscribed to the job channel.
    """
    while True:
        message: dict[str, Any] | None = await pubsub.get_message(
            ignore_subscribe_messages=True,
            timeout=1.0,
        )
        if message is None:
            # No message within the timeout window; yield to the event loop and
            # keep listening so the task remains cancellable.
            await asyncio.sleep(0)
            continue
        if message.get("type") != "message":
            continue

        data: Any = message.get("data")
        if data is None:
            continue
        if isinstance(data, (bytes, bytearray)):
            data = data.decode("utf-8")
        await websocket.send_text(str(data))


async def _keepalive(websocket: WebSocket) -> None:
    """Periodically send ping frames to keep the connection alive.

    Args:
        websocket: The accepted client WebSocket connection.
    """
    while True:
        await asyncio.sleep(KEEPALIVE_INTERVAL_SECONDS)
        await websocket.send_json({"type": "ping"})


@router.websocket("/ws/debate/{job_id}")
async def debate_stream(websocket: WebSocket, job_id: str) -> None:
    """Stream live debate events for ``job_id`` over a WebSocket connection.

    The endpoint accepts the socket, validates the job, replays the current job
    snapshot, then relays Redis pub/sub events while a concurrent task keeps the
    connection alive. All resources are released in a ``finally`` block on
    disconnect or error.

    Args:
        websocket: The incoming client WebSocket connection.
        job_id: Identifier of the job whose debate stream is requested.
    """
    await websocket.accept()

    job_service: JobService = websocket.app.state.job_service

    # Validate the job before subscribing so unknown jobs fail fast.
    try:
        job: JobResult = await job_service.get_job(job_id)
    except JobNotFoundError:
        logger.info("Rejecting debate stream for unknown job %s", job_id)
        await websocket.close(code=WS_CLOSE_JOB_NOT_FOUND)
        return

    # Replay the current state for late subscribers.
    try:
        await _send_snapshot(websocket, job)
    except WebSocketDisconnect:
        return

    pubsub: "aioredis.client.PubSub" = job_service.get_pubsub()
    channel: str = job_service.channel_for(job_id)

    relay_task: asyncio.Task[None] | None = None
    keepalive_task: asyncio.Task[None] | None = None

    try:
        await pubsub.subscribe(channel)

        relay_task = asyncio.create_task(_relay_messages(websocket, pubsub))
        keepalive_task = asyncio.create_task(_keepalive(websocket))

        # Run both tasks until one finishes (e.g. the client disconnects mid
        # send) or raises; gather propagates the first exception.
        await asyncio.gather(relay_task, keepalive_task)
    except WebSocketDisconnect:
        logger.info("Client disconnected from debate stream for job %s", job_id)
    except asyncio.CancelledError:  # pragma: no cover - server shutdown path
        logger.info("Debate stream task cancelled for job %s", job_id)
        raise
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("Error in debate stream for job %s", job_id)
    finally:
        # Cancel the sibling tasks so neither lingers after the connection ends.
        for task in (relay_task, keepalive_task):
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    # Suppress shutdown noise from cancelled/erroring tasks.
                    pass

        # Unsubscribe and close the pub/sub object cleanly.
        try:
            await pubsub.unsubscribe(channel)
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.debug("Failed to unsubscribe from %s", channel, exc_info=True)
        try:
            await pubsub.aclose()
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.debug("Failed to close pubsub for %s", channel, exc_info=True)
