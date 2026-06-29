"""The Argus Vision consensus pipeline orchestrator (23-dim numerical contract).

:class:`DebatePipeline` wires the ML stages into the end-to-end background job
that the API schedules:

1. Run both classifier agents (Agent A: EfficientNet-B4, Agent B: ViT-B/16).
2. Evaluate the debate trigger (JS divergence / entropy).
3. If triggered, compute spatial attention (Grad-CAM++ and attention rollout),
   the cross-agent disagreement map, and the contested bounding box. These power
   the UI heatmaps **and** the three spatial-attention features of the consensus
   vector.
4. Build the 23-dimensional consensus feature vector and run the calibrated
   consensus MLP to produce the final prediction.

The previous LLM (Groq) debate-text generation and 384-d sentence-embedding
fusion have been removed entirely. There is no Groq client, no argument
generation, and no debate transcript: the "debate" step is now purely numerical.
When the trigger does not fire, attention is skipped and the three attention
features are 0.0 (the fast path).

Every stage persists its result to Redis and publishes the matching WebSocket
event. All blocking work (torch inference, attention computation) is off-loaded
to worker threads with :func:`asyncio.to_thread`.
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
    AttentionResult,
    BoundingBox,
    ClassificationResult,
    ConsensusResult,
    TriggerResult,
)
from ml.agents.agent_a import AgentA
from ml.agents.agent_b import AgentB
from ml.attention.disagreement import compute_disagreement, extract_bbox
from ml.attention.gradcam import compute_gradcam_plusplus
from ml.attention.rollout import compute_attention_rollout
from ml.consensus.classifier import ConsensusClassifier
from ml.debate.trigger import evaluate_trigger
from services import image_service
from services.job_service import JobService

logger = logging.getLogger(__name__)

# ISIC-8 class names in their canonical (index 0..7) order.
CLASS_NAMES: list[str] = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]


class DebatePipeline:
    """Orchestrates the full consensus pipeline for a single image.

    Constructed once at application startup and reused across jobs. Construction
    loads both agents and the calibrated consensus head (with its feature
    scaler).

    Attributes:
        settings: The resolved application :class:`~core.config.Settings`.
        agent_a: The EfficientNet-B4 classifier (Agent A).
        agent_b: The ViT-B/16 classifier (Agent B).
        consensus: The calibrated 23-dim consensus fusion MLP.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Instantiate every ML component used across the pipeline's stages."""
        self.settings: Settings = settings if settings is not None else get_settings()
        self._loaded: bool = False

        checkpoint_dir = self.settings.MODEL_CHECKPOINT_DIR
        agent_a_ckpt = os.path.join(checkpoint_dir, self.settings.AGENT_A_CHECKPOINT)
        agent_b_ckpt = os.path.join(checkpoint_dir, self.settings.AGENT_B_CHECKPOINT)
        consensus_ckpt = os.path.join(checkpoint_dir, self.settings.CONSENSUS_CHECKPOINT)
        consensus_scaler = os.path.join(checkpoint_dir, self.settings.CONSENSUS_SCALER)
        pretrained_fallback = self.settings.PRETRAINED_FALLBACK

        self.agent_a: AgentA = AgentA(
            checkpoint_path=agent_a_ckpt,
            pretrained_fallback=pretrained_fallback,
        )
        self.agent_b: AgentB = AgentB(
            checkpoint_path=agent_b_ckpt,
            pretrained_fallback=pretrained_fallback,
        )
        self.consensus: ConsensusClassifier = ConsensusClassifier(
            checkpoint_path=consensus_ckpt,
            scaler_path=consensus_scaler,
        )

        self._loaded = True
        logger.info(
            "DebatePipeline ready (checkpoint_dir='%s'); 23-dim numerical "
            "consensus contract (no LLM debate).",
            checkpoint_dir,
        )

    def is_loaded(self) -> bool:
        """Report whether the pipeline finished loading its components."""
        return self._loaded

    async def run(
        self,
        job_id: str,
        image_path: str,
        job_service: JobService,
    ) -> None:
        """Execute the full consensus pipeline for one job, end to end.

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
                agent_id=self.agent_a.agent_id, result=result_a, heatmap_b64=None
            )
            agent_b_result = AgentResult(
                agent_id=self.agent_b.agent_id, result=result_b, heatmap_b64=None
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

            # Attention maps default to None (the consensus fast path -> the three
            # attention features are 0.0).
            heatmap_a: Optional[np.ndarray] = None
            heatmap_b: Optional[np.ndarray] = None

            if trigger.fired:
                # --- Stage 4: spatial attention analysis -------------------
                _attention, heatmap_a, heatmap_b = await self._compute_attention(
                    job_id=job_id,
                    job_service=job_service,
                    tensor=tensor,
                    original_pil=original_pil,
                    result_a=result_a,
                    result_b=result_b,
                    agent_a_result=agent_a_result,
                    agent_b_result=agent_b_result,
                )

            # --- Stage 5: consensus fusion on the 23-dim feature vector ------
            consensus: ConsensusResult = await asyncio.to_thread(
                self.consensus.predict,
                self._ordered_probs(result_a.probabilities),
                self._ordered_probs(result_b.probabilities),
                heatmap_a,
                heatmap_b,
            )

            await job_service.update_and_publish(
                job_id,
                {"type": "consensus_done", "result": consensus.model_dump()},
                status="consensus_done",
                debate=None,
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
        """Evaluate the debate trigger from both agents' distributions."""
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
    ) -> tuple[AttentionResult, np.ndarray, np.ndarray]:
        """Compute, persist and publish the spatial attention analysis.

        Returns:
            A tuple ``(attention_result, heatmap_a, heatmap_b)`` where the two
            heatmaps are the raw ``224x224`` ``[0, 1]`` saliency maps used both
            for the UI overlays and the three spatial-attention consensus
            features.
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

        heatmap_a_b64: str = await asyncio.to_thread(
            image_service.heatmap_to_b64, heatmap_a, original_pil
        )
        heatmap_b_b64: str = await asyncio.to_thread(
            image_service.heatmap_to_b64, heatmap_b, original_pil
        )

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

        m_delta, region_stats_a, region_stats_b = await asyncio.to_thread(
            compute_disagreement, heatmap_a, heatmap_b
        )
        bbox = await asyncio.to_thread(extract_bbox, m_delta)
        bbox = self._project_bbox_to_original_space(bbox, original_pil.size)
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

        return attention, heatmap_a, heatmap_b

    @staticmethod
    def _project_bbox_to_original_space(bbox: BoundingBox, original_size: tuple[int, int]) -> BoundingBox:
        """Map a 224x224 crop-space bbox back onto the original image."""
        orig_w, orig_h = original_size
        if orig_w <= 0 or orig_h <= 0:
            return bbox

        target_size = 256.0
        crop_size = 224.0
        scale = target_size / float(min(orig_w, orig_h))
        resized_w = float(orig_w) * scale
        resized_h = float(orig_h) * scale
        offset_x = max(0.0, (resized_w - crop_size) * 0.5)
        offset_y = max(0.0, (resized_h - crop_size) * 0.5)

        def _clamp(value: float, upper: int) -> int:
            return int(min(max(round(value), 0), upper))

        x1 = _clamp((bbox.x1 + offset_x) / scale, orig_w - 1)
        y1 = _clamp((bbox.y1 + offset_y) / scale, orig_h - 1)
        x2 = _clamp((bbox.x2 + offset_x) / scale, orig_w - 1)
        y2 = _clamp((bbox.y2 + offset_y) / scale, orig_h - 1)

        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    @staticmethod
    def _ordered_probs(probabilities: dict[str, float]) -> list[float]:
        """Flatten a class-probability mapping into canonical ISIC-8 order."""
        return [float(probabilities.get(name, 0.0)) for name in CLASS_NAMES]
