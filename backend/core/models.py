"""Pydantic data models for the Argus Vision backend.

This module is the single source of truth for the request/response payloads,
the persisted :class:`JobResult` document, and the WebSocket event union. Every
model here mirrors the shared contract exactly; the TypeScript frontend types
are kept in lock-step with these definitions.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Status type alias
# ---------------------------------------------------------------------------

JobStatus = Literal[
    "queued",
    "running",
    "agents_done",
    "trigger_evaluated",
    "attention_computed",
    "debate_round_1",
    "debate_round_2",
    "consensus_done",
    "failed",
]
"""Lifecycle status of a classification/debate job."""


# ---------------------------------------------------------------------------
# Core result models
# ---------------------------------------------------------------------------


class ClassificationResult(BaseModel):
    """A single classifier's prediction over the ISIC-8 label set.

    Attributes:
        pred_class: The predicted class label (one of the ISIC-8 classes).
        confidence: Softmax probability assigned to ``pred_class``.
        probabilities: Full probability distribution keyed by class label.
    """

    model_config = ConfigDict(populate_by_name=True)

    pred_class: str
    confidence: float
    probabilities: dict[str, float]


class AgentResult(BaseModel):
    """The output of one debating agent (Agent A or Agent B).

    Attributes:
        agent_id: Identifier of the agent (``"A"`` or ``"B"``).
        result: The agent's classification result.
        heatmap_b64: Base64-encoded saliency heatmap PNG, or ``None`` when no
            heatmap is available.
    """

    model_config = ConfigDict(populate_by_name=True)

    agent_id: str
    result: ClassificationResult
    heatmap_b64: Optional[str] = None


class TriggerResult(BaseModel):
    """Outcome of evaluating whether a debate should be triggered.

    Attributes:
        fired: Whether the debate was triggered.
        js_divergence: Jensen-Shannon divergence between the two agents'
            probability distributions.
        entropy_a: Entropy of Agent A's distribution.
        entropy_b: Entropy of Agent B's distribution.
        threshold_js: JS-divergence threshold applied for this evaluation.
        threshold_entropy: Entropy threshold applied for this evaluation.
    """

    model_config = ConfigDict(populate_by_name=True)

    fired: bool
    js_divergence: float
    entropy_a: float
    entropy_b: float
    threshold_js: float
    threshold_entropy: float


class BoundingBox(BaseModel):
    """An axis-aligned bounding box in pixel coordinates.

    Attributes:
        x1: Left edge x-coordinate.
        y1: Top edge y-coordinate.
        x2: Right edge x-coordinate.
        y2: Bottom edge y-coordinate.
    """

    model_config = ConfigDict(populate_by_name=True)

    x1: int
    y1: int
    x2: int
    y2: int


class AttentionResult(BaseModel):
    """Spatial attention analysis comparing the two agents.

    Attributes:
        heatmap_a_b64: Base64-encoded heatmap PNG for Agent A.
        heatmap_b_b64: Base64-encoded heatmap PNG for Agent B.
        disagreement_b64: Base64-encoded disagreement-map PNG.
        bbox: Bounding box of the most contested region.
        region_stats_a: Statistics for Agent A over the contested region.
        region_stats_b: Statistics for Agent B over the contested region.
    """

    model_config = ConfigDict(populate_by_name=True)

    heatmap_a_b64: str
    heatmap_b_b64: str
    disagreement_b64: str
    bbox: BoundingBox
    region_stats_a: dict[str, float]
    region_stats_b: dict[str, float]


class ArgumentResult(BaseModel):
    """An argument produced by one agent during a debate round.

    Attributes:
        agent_id: Identifier of the arguing agent (``"A"`` or ``"B"``).
        argument: The natural-language argument text.
        embedding: Sentence embedding of the argument.
        updated_probs: The agent's probability distribution after arguing.
    """

    model_config = ConfigDict(populate_by_name=True)

    agent_id: str
    argument: str
    embedding: list[float]
    updated_probs: dict[str, float]


class DebateRound(BaseModel):
    """A single round of debate containing both agents' arguments.

    Attributes:
        argument_a: Agent A's argument for this round.
        argument_b: Agent B's argument for this round.
    """

    model_config = ConfigDict(populate_by_name=True)

    argument_a: ArgumentResult
    argument_b: ArgumentResult


class ConsensusResult(BaseModel):
    """Calibrated consensus prediction produced by the consensus MLP.

    Attributes:
        pred_class: The final predicted class label.
        confidence: Calibrated probability assigned to ``pred_class``.
        probabilities: Full calibrated probability distribution by class.
        temperature: Learned temperature scalar used for calibration.
        ece: Expected calibration error of the consensus prediction.
    """

    model_config = ConfigDict(populate_by_name=True)

    pred_class: str
    confidence: float
    probabilities: dict[str, float]
    temperature: float
    ece: float


class JobResult(BaseModel):
    """The full, persisted state of a classification/debate job.

    Each optional stage field is populated as the pipeline progresses. The
    document is stored in Redis as a JSON string under ``argus:job:{job_id}``.

    Attributes:
        job_id: Unique identifier of the job.
        status: Current lifecycle status.
        created_at: Timestamp when the job was created.
        updated_at: Timestamp of the most recent update.
        agent_a: Agent A's result, once computed.
        agent_b: Agent B's result, once computed.
        trigger: Debate-trigger evaluation result, once computed.
        attention: Spatial attention analysis, once computed.
        debate: Debate round results, once computed.
        consensus: Final consensus result, once computed.
        error: Error message if the job failed, otherwise ``None``.
    """

    model_config = ConfigDict(populate_by_name=True)

    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    agent_a: Optional[AgentResult] = None
    agent_b: Optional[AgentResult] = None
    trigger: Optional[TriggerResult] = None
    attention: Optional[AttentionResult] = None
    debate: Optional[DebateRound] = None
    consensus: Optional[ConsensusResult] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# WebSocket event models (discriminated union)
# ---------------------------------------------------------------------------


class AgentsRunningEvent(BaseModel):
    """Event emitted when both agents begin inference.

    Attributes:
        type: Discriminator literal, always ``"agents_running"``.
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["agents_running"] = "agents_running"


class AgentsDoneEvent(BaseModel):
    """Event emitted when both agents have finished inference.

    Attributes:
        type: Discriminator literal, always ``"agents_done"``.
        agent_a: Agent A's completed result.
        agent_b: Agent B's completed result.
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["agents_done"] = "agents_done"
    agent_a: AgentResult
    agent_b: AgentResult


class TriggerEvaluatedEvent(BaseModel):
    """Event emitted after the debate trigger is evaluated.

    Attributes:
        type: Discriminator literal, always ``"trigger_evaluated"``.
        result: The trigger evaluation result.
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["trigger_evaluated"] = "trigger_evaluated"
    result: TriggerResult


class AttentionComputedEvent(BaseModel):
    """Event emitted after spatial attention analysis completes.

    Attributes:
        type: Discriminator literal, always ``"attention_computed"``.
        result: The attention analysis result.
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["attention_computed"] = "attention_computed"
    result: AttentionResult


class ArgumentTokenEvent(BaseModel):
    """Event emitted for each streamed token of an agent's argument.

    Attributes:
        type: Discriminator literal, always ``"argument_token"``.
        agent: Which agent is speaking (``"A"`` or ``"B"``).
        token: The streamed token text.
        round: Debate round number (``1`` or ``2``).
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["argument_token"] = "argument_token"
    agent: Literal["A", "B"]
    token: str
    round: int


class ArgumentDoneEvent(BaseModel):
    """Event emitted when an agent has finished its argument.

    Attributes:
        type: Discriminator literal, always ``"argument_done"``.
        agent: Which agent finished (``"A"`` or ``"B"``).
        argument: The complete argument text.
        round: Debate round number (``1`` or ``2``).
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["argument_done"] = "argument_done"
    agent: Literal["A", "B"]
    argument: str
    round: int


class ConsensusDoneEvent(BaseModel):
    """Event emitted when the consensus result is ready.

    Attributes:
        type: Discriminator literal, always ``"consensus_done"``.
        result: The final consensus result.
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["consensus_done"] = "consensus_done"
    result: ConsensusResult


class ErrorEvent(BaseModel):
    """Event emitted when the pipeline encounters an error.

    Attributes:
        type: Discriminator literal, always ``"error"``.
        message: Human-readable error description.
    """

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["error"] = "error"
    message: str


DebateEvent = Annotated[
    Union[
        AgentsRunningEvent,
        AgentsDoneEvent,
        TriggerEvaluatedEvent,
        AttentionComputedEvent,
        ArgumentTokenEvent,
        ArgumentDoneEvent,
        ConsensusDoneEvent,
        ErrorEvent,
    ],
    Field(discriminator="type"),
]
"""Discriminated union of all WebSocket event payloads, keyed on ``type``."""
