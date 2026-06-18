"""Redis-backed job persistence and event publishing for Argus Vision.

This module exposes :class:`JobService`, an asynchronous service that owns the
lifecycle of a classification job. Jobs are serialized as JSON
(:class:`core.models.JobResult`) and stored in Redis under the key
``argus:job:{job_id}`` with a 3600 second TTL. The originating image path is
stored alongside under ``argus:img:{job_id}``. WebSocket-bound debate events are
published to the pub/sub channel ``argus:debate:{job_id}``.

The service uses the redis-py 5.x asyncio API (``from redis import asyncio as
aioredis``); the legacy standalone ``aioredis`` package is never imported.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from redis import asyncio as aioredis

from core.config import get_settings
from core.exceptions import JobNotFoundError
from core.models import (
    AgentResult,
    AttentionResult,
    ConsensusResult,
    DebateRound,
    JobResult,
    TriggerResult,
)

# Re-exported for backwards compatibility so that
# ``from services.job_service import JobNotFoundError`` continues to resolve to
# the canonical :class:`core.exceptions.JobNotFoundError` (an ``ArgusError`` with
# ``status_code=404``), which the FastAPI handler in ``main.py`` translates into
# a contract-required HTTP 404 response.
__all__ = ["JobNotFoundError", "JobService"]

#: Redis key prefix for the serialized :class:`JobResult` JSON value.
JOB_KEY_PREFIX = "argus:job:"
#: Redis key prefix for the parallel image-path value associated with a job.
IMG_KEY_PREFIX = "argus:img:"
#: Pub/sub channel prefix used to relay debate events to WebSocket clients.
CHANNEL_PREFIX = "argus:debate:"
#: Time-to-live (in seconds) applied to job and image keys.
JOB_TTL_SECONDS = 3600


class JobService:
    """Asynchronous Redis-backed store for :class:`JobResult` jobs.

    The Redis client is created lazily on first use so that constructing a
    :class:`JobService` never performs I/O. All public methods are coroutines.
    """

    def __init__(self, redis_url: str | None = None) -> None:
        """Initialise the service without opening a connection.

        Args:
            redis_url: Optional Redis connection URL. When omitted, the value
                of ``REDIS_URL`` from application settings is used.
        """
        self._redis_url: str = redis_url if redis_url is not None else get_settings().REDIS_URL
        self._client: aioredis.Redis | None = None

    @property
    def client(self) -> aioredis.Redis:
        """Return the lazily-instantiated, decode-enabled Redis client.

        Returns:
            A connected :class:`redis.asyncio.Redis` instance with
            ``decode_responses=True`` so values round-trip as ``str``.
        """
        if self._client is None:
            self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._client

    @staticmethod
    def _job_key(job_id: str) -> str:
        """Build the Redis key holding the serialized job for ``job_id``."""
        return f"{JOB_KEY_PREFIX}{job_id}"

    @staticmethod
    def _img_key(job_id: str) -> str:
        """Build the Redis key holding the image path for ``job_id``."""
        return f"{IMG_KEY_PREFIX}{job_id}"

    @staticmethod
    def _channel(job_id: str) -> str:
        """Build the pub/sub channel name used to relay events for ``job_id``."""
        return f"{CHANNEL_PREFIX}{job_id}"

    async def ping(self) -> bool:
        """Check Redis connectivity.

        Returns:
            ``True`` if the server responded to ``PING``; otherwise ``False``.
        """
        try:
            return bool(await self.client.ping())
        except Exception:
            return False

    async def create_job(self, job_id: str, image_path: str) -> JobResult:
        """Create and persist a fresh ``queued`` job.

        A new :class:`JobResult` is built with ``status="queued"`` and matching
        ``created_at``/``updated_at`` timestamps, then stored as JSON with a
        3600 second TTL. The originating image path is stored in a parallel key.

        Args:
            job_id: Unique identifier for the new job.
            image_path: Absolute filesystem path to the uploaded image.

        Returns:
            The newly created :class:`JobResult`.
        """
        now = datetime.utcnow()
        job = JobResult(
            job_id=job_id,
            status="queued",
            created_at=now,
            updated_at=now,
            agent_a=None,
            agent_b=None,
            trigger=None,
            attention=None,
            debate=None,
            consensus=None,
            error=None,
        )
        await self.client.set(
            self._job_key(job_id),
            job.model_dump_json(),
            ex=JOB_TTL_SECONDS,
        )
        await self.client.set(
            self._img_key(job_id),
            image_path,
            ex=JOB_TTL_SECONDS,
        )
        return job

    async def get_job(self, job_id: str) -> JobResult:
        """Load and deserialize the job for ``job_id``.

        Args:
            job_id: Identifier of the job to fetch.

        Returns:
            The stored :class:`JobResult`.

        Raises:
            JobNotFoundError: If no job exists for ``job_id``.
        """
        raw = await self.client.get(self._job_key(job_id))
        if raw is None:
            raise JobNotFoundError(f"Job not found: {job_id}")
        return JobResult.model_validate_json(raw)

    async def get_image_path(self, job_id: str) -> str:
        """Return the stored image path for ``job_id``.

        Args:
            job_id: Identifier of the job whose image path is requested.

        Returns:
            The absolute image path recorded at creation time.

        Raises:
            JobNotFoundError: If no image path exists for ``job_id``.
        """
        raw = await self.client.get(self._img_key(job_id))
        if raw is None:
            raise JobNotFoundError(f"Job not found: {job_id}")
        return raw

    async def update_job(self, job_id: str, **fields: Any) -> JobResult:
        """Apply a partial update to an existing job and re-persist it.

        The existing job JSON is loaded, the supplied recognised fields are
        applied, ``updated_at`` is refreshed to the current UTC time, and the
        job is written back with its TTL refreshed to 3600 seconds.

        Recognised fields: ``status``, ``agent_a``, ``agent_b``, ``trigger``,
        ``attention``, ``debate``, ``consensus``, ``error``. Each may be passed
        either as the corresponding pydantic model instance or as a plain dict
        (which is validated into the model). Unknown fields are ignored.

        Args:
            job_id: Identifier of the job to mutate.
            **fields: Field values to overwrite on the stored job.

        Returns:
            The updated :class:`JobResult`.

        Raises:
            JobNotFoundError: If no job exists for ``job_id``.
        """
        job = await self.get_job(job_id)

        if "status" in fields and fields["status"] is not None:
            job.status = fields["status"]
        if "agent_a" in fields:
            job.agent_a = self._coerce(fields["agent_a"], AgentResult)
        if "agent_b" in fields:
            job.agent_b = self._coerce(fields["agent_b"], AgentResult)
        if "trigger" in fields:
            job.trigger = self._coerce(fields["trigger"], TriggerResult)
        if "attention" in fields:
            job.attention = self._coerce(fields["attention"], AttentionResult)
        if "debate" in fields:
            job.debate = self._coerce(fields["debate"], DebateRound)
        if "consensus" in fields:
            job.consensus = self._coerce(fields["consensus"], ConsensusResult)
        if "error" in fields:
            job.error = fields["error"]

        job.updated_at = datetime.utcnow()

        await self.client.set(
            self._job_key(job_id),
            job.model_dump_json(),
            ex=JOB_TTL_SECONDS,
        )
        # Refresh the parallel image-path TTL so it expires in lockstep.
        await self.client.expire(self._img_key(job_id), JOB_TTL_SECONDS)
        return job

    @staticmethod
    def _coerce(value: Any, model: type) -> Any:
        """Coerce ``value`` into ``model`` when given a raw mapping.

        Args:
            value: Either ``None``, a ``model`` instance, or a ``dict`` to
                validate into ``model``.
            model: The target pydantic model class.

        Returns:
            ``None``, the original model instance, or a validated model.
        """
        if value is None:
            return None
        if isinstance(value, model):
            return value
        if isinstance(value, dict):
            return model.model_validate(value)
        return value

    async def publish_event(self, job_id: str, event: dict[str, Any]) -> int:
        """Publish a single WebSocket event to the job's pub/sub channel.

        Args:
            job_id: Identifier of the job the event belongs to.
            event: The event payload; serialized with ``json.dumps`` using
                ``default=str`` so datetimes and other objects encode safely.

        Returns:
            The number of subscribers that received the message.
        """
        payload = json.dumps(event, default=str)
        return int(await self.client.publish(self._channel(job_id), payload))

    async def update_and_publish(
        self,
        job_id: str,
        event: dict[str, Any],
        **fields: Any,
    ) -> JobResult:
        """Persist a partial job update and publish a WebSocket event atomically.

        This is the primary helper used by the debate pipeline: it both advances
        the persisted job state and notifies connected clients in one call.

        Args:
            job_id: Identifier of the job to mutate and notify on.
            event: The WebSocket event payload to publish.
            **fields: Field values to overwrite on the stored job.

        Returns:
            The updated :class:`JobResult`.

        Raises:
            JobNotFoundError: If no job exists for ``job_id``.
        """
        job = await self.update_job(job_id, **fields)
        await self.publish_event(job_id, event)
        return job

    async def delete_job(self, job_id: str) -> bool:
        """Delete both the job and image keys for ``job_id``.

        Args:
            job_id: Identifier of the job to remove.

        Returns:
            ``True`` once the delete has been issued (idempotent).
        """
        await self.client.delete(self._job_key(job_id), self._img_key(job_id))
        return True

    def get_pubsub(self) -> aioredis.client.PubSub:
        """Return a fresh pub/sub object for the WebSocket relay.

        The caller is responsible for subscribing to the appropriate channel
        (see :meth:`channel_for`) and for closing the pub/sub when finished.

        Returns:
            A :class:`redis.asyncio.client.PubSub` bound to this client.
        """
        return self.client.pubsub()

    def channel_for(self, job_id: str) -> str:
        """Return the pub/sub channel name for ``job_id``.

        Args:
            job_id: Identifier of the job whose channel is requested.

        Returns:
            The fully-qualified channel name, e.g. ``argus:debate:<id>``.
        """
        return self._channel(job_id)

    async def close(self) -> None:
        """Close the underlying Redis client and release its connection pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
