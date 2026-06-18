"""The Argus Vision adversarial debate pipeline orchestrator.

:class:`DebatePipeline` wires together every ML stage into the end-to-end job
that the API schedules as a background task:

1. Run both classifier agents (Agent A: EfficientNet-B4, Agent B: ViT-B/16).
2. Evaluate the debate trigger (JS divergence / entropy).
3. If triggered, compute spatial attention (Grad-CAM++ and attention rollout),
   the cross-agent disagreement map, and the contested bounding box.
4. If triggered, run a two-round LLM debate, streaming each argument token over
   the WebSocket relay.
5. Fuse every signal with the calibrated consensus MLP to produce the final
   prediction.

Every stage persists its result to Redis and publishes the matching WebSocket
event through :class:`services.job_service.JobService`. All blocking work (torch
inference, Groq calls, sentence-transformer encoding) is off-loaded to worker
threads with :func:`asyncio.to_thread` so the event loop is never blocked. The
entire run is wrapped so that any failure is reported as an ``error`` event and a
``failed`` job status rather than crashing the background task silently.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image

from core.config import Settings, get_settings
from core.models import (
    AgentResult,
    ArgumentResult,
    AttentionResult,
    ClassificationResult,
    ConsensusResult,
    DebateRound,
    TriggerResult,
)
from ml.agents.agent_a import AgentA
from ml.agents.agent_b import AgentB
from ml.attention.disagreement import compute_disagreement, extract_bbox
from ml.attention.gradcam import compute_gradcam_plusplus
from ml.attention.rollout import compute_attention_rollout
from ml.consensus.classifier import ConsensusClassifier
from ml.debate.argument_gen import (
    build_prompt,
    generate_argument,
    generate_counter_argument,
)
from ml.debate.encoder import ArgumentEncoder
from ml.debate.trigger import evaluate_trigger
from services import image_service
from services.job_service import JobService

logger = logging.getLogger(__name__)

# ISIC-8 class names in their canonical (index 0..7) order.
CLASS_NAMES: list[str] = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

# Dimensionality of an argument sentence embedding (all-MiniLM-L6-v2).
EMBEDDING_DIM: int = 384

# Per-token streaming delay (seconds) so the WebSocket client renders a natural
# typewriter cadence during the debate.
_TOKEN_DELAY_SECONDS: float = 0.05


class DebatePipeline:
    """Orchestrates the full multi-agent visual debate for a single image.

    The pipeline is constructed once at application startup and reused across
    every job. Constructing it loads both agents, the consensus head and the
    argument encoder, and creates a Groq client when an API key is configured.

    Attributes:
        settings: The resolved application :class:`~core.config.Settings`.
        agent_a: The EfficientNet-B4 classifier (Agent A).
        agent_b: The ViT-B/16 classifier (Agent B).
        encoder: The sentence encoder producing 384-d argument embeddings.
        consensus: The calibrated consensus fusion MLP.
        groq_client: An initialised Groq SDK client, or ``None`` when no key is
            configured (the debate then uses deterministic fallback arguments).
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Instantiate every ML component used across the pipeline's stages.

        Args:
            settings: Application settings. When ``None`` the process-wide cached
                :func:`~core.config.get_settings` instance is used so the
                pipeline can be constructed with no arguments at startup.
        """
        self.settings: Settings = settings if settings is not None else get_settings()
        self._loaded: bool = False

        checkpoint_dir = self.settings.MODEL_CHECKPOINT_DIR
        agent_a_ckpt = os.path.join(checkpoint_dir, self.settings.AGENT_A_CHECKPOINT)
        agent_b_ckpt = os.path.join(checkpoint_dir, self.settings.AGENT_B_CHECKPOINT)
        consensus_ckpt = os.path.join(checkpoint_dir, self.settings.CONSENSUS_CHECKPOINT)
        pretrained_fallback = self.settings.PRETRAINED_FALLBACK

        self.agent_a: AgentA = AgentA(
            checkpoint_path=agent_a_ckpt,
            pretrained_fallback=pretrained_fallback,
        )
        self.agent_b: AgentB = AgentB(
            checkpoint_path=agent_b_ckpt,
            pretrained_fallback=pretrained_fallback,
        )
        self.encoder: ArgumentEncoder = ArgumentEncoder()
        self.consensus: ConsensusClassifier = ConsensusClassifier(
            checkpoint_path=consensus_ckpt,
        )

        self.groq_client = self._build_groq_client(self.settings.GROQ_API_KEY)

        self._loaded = True
        logger.info(
            "DebatePipeline ready (groq_enabled=%s, checkpoint_dir='%s').",
            self.groq_client is not None,
            checkpoint_dir,
        )

    @staticmethod
    def _build_groq_client(api_key: str) -> Optional[object]:
        """Create a Groq client when an API key is configured.

        Args:
            api_key: The Groq API key from settings (possibly empty).

        Returns:
            An initialised ``groq.Groq`` client, or ``None`` when no key is set or
            the client could not be constructed (the debate then degrades to
            deterministic fallback arguments).
        """
        if not api_key:
            logger.warning(
                "No GROQ_API_KEY configured; debate arguments will use the "
                "deterministic offline fallback."
            )
            return None
        try:
            import groq

            return groq.Groq(api_key=api_key)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully on any error.
            logger.warning(
                "Failed to initialise Groq client (%s); debate arguments will "
                "use the deterministic offline fallback.",
                exc,
            )
            return None

    def is_loaded(self) -> bool:
        """Report whether the pipeline finished loading its components.

        Returns:
            ``True`` once construction completed successfully, else ``False``.
        """
        return self._loaded

    async def run(
        self,
        job_id: str,
        image_path: str,
        job_service: JobService,
    ) -> None:
        """Execute the full debate pipeline for one job, end to end.

        Each stage persists its result and publishes a WebSocket event. The whole
        body is wrapped so that any exception is logged and surfaced as an
        ``error`` event with a ``failed`` job status, guaranteeing the background
        task never fails silently.

        Args:
            job_id: Identifier of the job being processed.
            image_path: Filesystem path to the uploaded image to classify.
            job_service: The Redis-backed store used to persist state and publish
                events.
        """
        try:
            # --- Stage 1: agents start -------------------------------------
            await job_service.update_and_publish(
                job_id,
                {"type": "agents_running"},
                status="running",
            )

            tensor, original_pil = image_service.preprocess_image(image_path)

            # --- Stage 2: run both agents ----------------------------------
            result_a = await asyncio.to_thread(self.agent_a.predict, tensor)
            result_b = await asyncio.to_thread(self.agent_b.predict, tensor)

            agent_a_result = AgentResult(
                agent_id=self.agent_a.agent_id,
                result=result_a,
                heatmap_b64=None,
            )
            agent_b_result = AgentResult(
                agent_id=self.agent_b.agent_id,
                result=result_b,
                heatmap_b64=None,
            )

            await job_service.update_and_publish(
                job_id,
                {
                    "type": "agents_done",
                    "agent_a": agent_a_result.model_dump(),
                    "agent_b": agent_b_result.model_dump(),
                },
                status="agents_done",
                agent_a=agent_a_result,
                agent_b=agent_b_result,
            )

            # --- Stage 3: evaluate the debate trigger ----------------------
            trigger: TriggerResult = await asyncio.to_thread(
                self._evaluate_trigger,
                result_a.probabilities,
                result_b.probabilities,
            )
            await job_service.update_and_publish(
                job_id,
                {"type": "trigger_evaluated", "result": trigger.model_dump()},
                status="trigger_evaluated",
                trigger=trigger,
            )

            # Defaults for the consensus fast path (no debate).
            spatial_stats: list[float] = [0.0, 0.0, 0.0, 0.0]
            embedding_a: list[float] = [0.0] * EMBEDDING_DIM
            embedding_b: list[float] = [0.0] * EMBEDDING_DIM
            debate_round: Optional[DebateRound] = None

            if trigger.fired:
                # --- Stage 4: spatial attention analysis -------------------
                attention, spatial_stats = await self._compute_attention(
                    job_id=job_id,
                    job_service=job_service,
                    tensor=tensor,
                    original_pil=original_pil,
                    result_a=result_a,
                    result_b=result_b,
                    agent_a_result=agent_a_result,
                    agent_b_result=agent_b_result,
                )

                # --- Stages 5-7: two-round debate + embeddings -------------
                debate_round, embedding_a, embedding_b = await self._run_debate(
                    job_id=job_id,
                    job_service=job_service,
                    result_a=result_a,
                    result_b=result_b,
                    attention=attention,
                )

            # --- Stage 7/8: consensus fusion (always runs) -----------------
            consensus: ConsensusResult = await asyncio.to_thread(
                self.consensus.predict,
                list(self._ordered_probs(result_a.probabilities)),
                list(self._ordered_probs(result_b.probabilities)),
                spatial_stats,
                embedding_a,
                embedding_b,
            )

            await job_service.update_and_publish(
                job_id,
                {"type": "consensus_done", "result": consensus.model_dump()},
                status="consensus_done",
                debate=debate_round,
                consensus=consensus,
            )
        except Exception as exc:  # noqa: BLE001 - report any failure to the client.
            logger.error(
                "DebatePipeline failed for job '%s': %s", job_id, exc, exc_info=True
            )
            try:
                await job_service.update_and_publish(
                    job_id,
                    {"type": "error", "message": str(exc)},
                    status="failed",
                    error=str(exc),
                )
            except Exception as publish_exc:  # noqa: BLE001 - best-effort reporting.
                logger.error(
                    "Failed to publish error event for job '%s': %s",
                    job_id,
                    publish_exc,
                    exc_info=True,
                )

    def _evaluate_trigger(
        self,
        probs_a: dict[str, float],
        probs_b: dict[str, float],
    ) -> TriggerResult:
        """Evaluate the debate trigger from both agents' distributions.

        Args:
            probs_a: Agent A's probability mapping.
            probs_b: Agent B's probability mapping.

        Returns:
            The :class:`~core.models.TriggerResult` for this job.
        """
        return evaluate_trigger(
            probs_a,
            probs_b,
            self.settings.DEBATE_JS_THRESHOLD,
            self.settings.DEBATE_ENTROPY_THRESHOLD,
        )

    async def _compute_attention(
        self,
        job_id: str,
        job_service: JobService,
        tensor: torch.Tensor,
        original_pil: Image.Image,
        result_a: ClassificationResult,
        result_b: ClassificationResult,
        agent_a_result: AgentResult,
        agent_b_result: AgentResult,
    ) -> tuple[AttentionResult, list[float]]:
        """Compute, persist and publish the spatial attention analysis.

        Agent A's Grad-CAM++ (on its predicted class) and Agent B's attention
        rollout are computed in worker threads, rendered into base64 overlays and
        attached to the agents. The disagreement map and contested bounding box
        are derived, an :class:`~core.models.AttentionResult` is persisted and the
        ``attention_computed`` event is published. The ``agents_done`` event is
        re-published so the client receives the now-populated heatmaps.

        Args:
            job_id: Identifier of the job being processed.
            job_service: The Redis-backed store used to persist and publish.
            tensor: The pre-processed input batch (shape ``[1, 3, 224, 224]``).
            original_pil: The original RGB PIL image used for heatmap overlays.
            result_a: Agent A's :class:`~core.models.ClassificationResult`.
            result_b: Agent B's :class:`~core.models.ClassificationResult`.
            agent_a_result: Agent A's :class:`~core.models.AgentResult` to update
                in place with its heatmap.
            agent_b_result: Agent B's :class:`~core.models.AgentResult` to update
                in place with its heatmap.

        Returns:
            A tuple of the persisted :class:`~core.models.AttentionResult` and the
            four-element ``spatial_stats`` list ``[mean_a, mean_b, std_a, std_b]``.
        """
        target_class = CLASS_NAMES.index(result_a.pred_class)

        heatmap_a: np.ndarray = await asyncio.to_thread(
            compute_gradcam_plusplus,
            self.agent_a.get_model(),
            tensor.to(self.agent_a.device),
            target_class,
        )
        heatmap_b: np.ndarray = await asyncio.to_thread(
            compute_attention_rollout,
            self.agent_b.get_model(),
            tensor.to(self.agent_b.device),
        )

        # Render base64 overlays for each agent's saliency map.
        heatmap_a_b64: str = await asyncio.to_thread(
            image_service.heatmap_to_b64, heatmap_a, original_pil
        )
        heatmap_b_b64: str = await asyncio.to_thread(
            image_service.heatmap_to_b64, heatmap_b, original_pil
        )

        # Attach the overlays to the agent results and re-publish agents_done so
        # the client picks up the now-available heatmaps.
        agent_a_result.heatmap_b64 = heatmap_a_b64
        agent_b_result.heatmap_b64 = heatmap_b_b64
        await job_service.update_and_publish(
            job_id,
            {
                "type": "agents_done",
                "agent_a": agent_a_result.model_dump(),
                "agent_b": agent_b_result.model_dump(),
            },
            agent_a=agent_a_result,
            agent_b=agent_b_result,
        )

        # Disagreement map, contested-region statistics and bounding box.
        m_delta, region_stats_a, region_stats_b = await asyncio.to_thread(
            compute_disagreement, heatmap_a, heatmap_b
        )
        bbox = await asyncio.to_thread(extract_bbox, m_delta)
        disagreement_b64: str = await asyncio.to_thread(
            image_service.array_to_b64, m_delta
        )

        attention = AttentionResult(
            heatmap_a_b64=heatmap_a_b64,
            heatmap_b_b64=heatmap_b_b64,
            disagreement_b64=disagreement_b64,
            bbox=bbox,
            region_stats_a=region_stats_a,
            region_stats_b=region_stats_b,
        )

        await job_service.update_and_publish(
            job_id,
            {"type": "attention_computed", "result": attention.model_dump()},
            status="attention_computed",
            attention=attention,
        )

        spatial_stats: list[float] = [
            float(region_stats_a.get("mean", 0.0)),
            float(region_stats_b.get("mean", 0.0)),
            float(region_stats_a.get("std", 0.0)),
            float(region_stats_b.get("std", 0.0)),
        ]
        return attention, spatial_stats

    async def _run_debate(
        self,
        job_id: str,
        job_service: JobService,
        result_a: ClassificationResult,
        result_b: ClassificationResult,
        attention: AttentionResult,
    ) -> tuple[DebateRound, list[float], list[float]]:
        """Run the two-round adversarial debate and encode the final arguments.

        Round 1 has each agent open with a defence of its prediction; round 2 has
        each agent rebut its opponent's opening argument, emitting a bounded
        confidence delta. Every argument is streamed token-by-token over the
        WebSocket relay. The final round-2 arguments are embedded and the agents'
        probabilities are nudged by their confidence deltas and renormalised.

        Args:
            job_id: Identifier of the job being processed.
            job_service: The Redis-backed store used to persist and publish.
            result_a: Agent A's :class:`~core.models.ClassificationResult`.
            result_b: Agent B's :class:`~core.models.ClassificationResult`.
            attention: The computed attention analysis (region stats + bbox).

        Returns:
            A tuple ``(debate_round, embedding_a, embedding_b)`` where
            ``debate_round`` is the persisted :class:`~core.models.DebateRound`
            and the embeddings are the 384-d encodings of the final arguments.
        """
        bbox = attention.bbox
        region_stats_a = attention.region_stats_a
        region_stats_b = attention.region_stats_b

        # --- Stage 5: round 1 opening arguments ------------------------------
        prompt_a = build_prompt(
            agent_id=self.agent_a.agent_id,
            pred_class=result_a.pred_class,
            confidence=result_a.confidence,
            region_stats=region_stats_a,
            bbox=bbox,
            opponent_pred=result_b.pred_class,
            opponent_conf=result_b.confidence,
            round=1,
        )
        prompt_b = build_prompt(
            agent_id=self.agent_b.agent_id,
            pred_class=result_b.pred_class,
            confidence=result_b.confidence,
            region_stats=region_stats_b,
            bbox=bbox,
            opponent_pred=result_a.pred_class,
            opponent_conf=result_a.confidence,
            round=1,
        )

        argument_a_r1: str = await asyncio.to_thread(
            generate_argument, prompt_a, self.groq_client, self.settings.GROQ_MODEL
        )
        await self._stream_argument(job_service, job_id, "A", argument_a_r1, 1)

        argument_b_r1: str = await asyncio.to_thread(
            generate_argument, prompt_b, self.groq_client, self.settings.GROQ_MODEL
        )
        await self._stream_argument(job_service, job_id, "B", argument_b_r1, 1)

        await job_service.update_job(job_id, status="debate_round_1")

        # --- Stage 6: round 2 counter-arguments ------------------------------
        counter_a, delta_a = await asyncio.to_thread(
            generate_counter_argument,
            argument_a_r1,
            argument_b_r1,
            result_a.pred_class,
            self.groq_client,
            self.settings.GROQ_MODEL,
            self.agent_a.agent_id,
            result_a.confidence,
            result_b.pred_class,
            result_b.confidence,
            region_stats_a,
            bbox,
        )
        await self._stream_argument(job_service, job_id, "A", counter_a, 2)

        counter_b, delta_b = await asyncio.to_thread(
            generate_counter_argument,
            argument_b_r1,
            argument_a_r1,
            result_b.pred_class,
            self.groq_client,
            self.settings.GROQ_MODEL,
            self.agent_b.agent_id,
            result_b.confidence,
            result_a.pred_class,
            result_a.confidence,
            region_stats_b,
            bbox,
        )
        await self._stream_argument(job_service, job_id, "B", counter_b, 2)

        await job_service.update_job(job_id, status="debate_round_2")

        # --- Stage 7: encode final arguments and build the debate round ------
        embedding_a: list[float] = await asyncio.to_thread(
            self.encoder.encode, counter_a
        )
        embedding_b: list[float] = await asyncio.to_thread(
            self.encoder.encode, counter_b
        )

        updated_probs_a = self._nudge_probabilities(
            result_a.probabilities, result_a.pred_class, delta_a
        )
        updated_probs_b = self._nudge_probabilities(
            result_b.probabilities, result_b.pred_class, delta_b
        )

        debate_round = DebateRound(
            argument_a=ArgumentResult(
                agent_id=self.agent_a.agent_id,
                argument=counter_a,
                embedding=embedding_a,
                updated_probs=updated_probs_a,
            ),
            argument_b=ArgumentResult(
                agent_id=self.agent_b.agent_id,
                argument=counter_b,
                embedding=embedding_b,
                updated_probs=updated_probs_b,
            ),
        )
        await job_service.update_job(job_id, debate=debate_round)

        return debate_round, embedding_a, embedding_b

    async def _stream_argument(
        self,
        job_service: JobService,
        job_id: str,
        agent: str,
        argument: str,
        round_number: int,
    ) -> None:
        """Stream an argument word-by-word, then publish the completed argument.

        The argument is split on whitespace; each word (with a trailing space) is
        published as an ``argument_token`` event followed by a short delay so the
        client renders a typewriter effect. After streaming, a single
        ``argument_done`` event carrying the full text is published.

        Args:
            job_service: The Redis-backed store used to publish events.
            job_id: Identifier of the job being processed.
            agent: Which agent is speaking (``"A"`` or ``"B"``).
            argument: The full argument text to stream.
            round_number: The debate round number (``1`` or ``2``).
        """
        for word in argument.split():
            await job_service.publish_event(
                job_id,
                {
                    "type": "argument_token",
                    "agent": agent,
                    "token": word + " ",
                    "round": round_number,
                },
            )
            await asyncio.sleep(_TOKEN_DELAY_SECONDS)

        await job_service.publish_event(
            job_id,
            {
                "type": "argument_done",
                "agent": agent,
                "argument": argument,
                "round": round_number,
            },
        )

    @staticmethod
    def _ordered_probs(probabilities: dict[str, float]) -> list[float]:
        """Flatten a class-probability mapping into canonical ISIC-8 order.

        Args:
            probabilities: A ``{class_code: probability}`` mapping.

        Returns:
            An 8-element list of probabilities ordered by :data:`CLASS_NAMES`
            (missing classes default to ``0.0``).
        """
        return [float(probabilities.get(name, 0.0)) for name in CLASS_NAMES]

    @staticmethod
    def _nudge_probabilities(
        probabilities: dict[str, float],
        pred_class: str,
        delta: float,
    ) -> dict[str, float]:
        """Adjust a distribution by a confidence delta on the predicted class.

        The predicted class's probability is shifted by ``delta`` (clamped into
        ``[0, 1]``) and the whole distribution is renormalised so it sums to one.
        A degenerate (all-zero) distribution is returned unchanged.

        Args:
            probabilities: The agent's original ``{class_code: probability}``
                mapping.
            pred_class: The class whose probability is nudged.
            delta: The signed confidence adjustment from round 2.

        Returns:
            A renormalised ``{class_code: probability}`` mapping over ISIC-8.
        """
        adjusted: dict[str, float] = {
            name: float(probabilities.get(name, 0.0)) for name in CLASS_NAMES
        }
        adjusted[pred_class] = float(
            min(1.0, max(0.0, adjusted.get(pred_class, 0.0) + delta))
        )

        total = sum(adjusted.values())
        if total <= 0.0:
            return adjusted
        return {name: value / total for name, value in adjusted.items()}
