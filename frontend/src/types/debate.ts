/**
 * TypeScript mirrors of the backend Pydantic schemas defined in
 * backend/core/models.py. Field names and types MUST match the backend
 * contract exactly so that JSON payloads (HTTP responses and WebSocket
 * events) deserialize without transformation on the frontend.
 */

/** Result of a single-model classification over the 8 ISIC classes. */
export interface ClassificationResult {
  pred_class: string;
  confidence: number;
  probabilities: Record<string, number>;
}

/** A single agent's classification output plus its optional heatmap. */
export interface AgentResult {
  agent_id: string;
  result: ClassificationResult;
  heatmap_b64: string | null;
}

/** Outcome of evaluating whether the adversarial debate should fire. */
export interface TriggerResult {
  fired: boolean;
  js_divergence: number;
  entropy_a: number;
  entropy_b: number;
  threshold_js: number;
  threshold_entropy: number;
}

/** Pixel-space bounding box of the contested region. */
export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** Spatial attention analysis for both agents and their disagreement map. */
export interface AttentionResult {
  heatmap_a_b64: string;
  heatmap_b_b64: string;
  disagreement_b64: string;
  bbox: BoundingBox;
  region_stats_a: Record<string, number>;
  region_stats_b: Record<string, number>;
}

/** A single agent's debate argument with its embedding and updated beliefs. */
export interface ArgumentResult {
  agent_id: string;
  argument: string;
  embedding: number[];
  updated_probs: Record<string, number>;
}

/** One full round of the debate: argument from A then from B. */
export interface DebateRound {
  argument_a: ArgumentResult;
  argument_b: ArgumentResult;
}

/** Final calibrated consensus produced by the consensus MLP. */
export interface ConsensusResult {
  pred_class: string;
  confidence: number;
  probabilities: Record<string, number>;
  temperature: number;
  ece: number;
}

/** The lifecycle status of a classification job. */
export type JobStatus =
  | "queued"
  | "running"
  | "agents_done"
  | "trigger_evaluated"
  | "attention_computed"
  | "debate_round_1"
  | "debate_round_2"
  | "consensus_done"
  | "failed";

/** The complete, accumulated state of a classification job. */
export interface JobResult {
  job_id: string;
  status: JobStatus;
  created_at: string;
  updated_at: string;
  agent_a: AgentResult | null;
  agent_b: AgentResult | null;
  trigger: TriggerResult | null;
  attention: AttentionResult | null;
  debate: DebateRound | null;
  consensus: ConsensusResult | null;
  error: string | null;
}

/** Emitted when both agents begin their independent inference. */
export interface AgentsRunningEvent {
  type: "agents_running";
}

/** Emitted once both agents have produced their classification results. */
export interface AgentsDoneEvent {
  type: "agents_done";
  agent_a: AgentResult;
  agent_b: AgentResult;
}

/** Emitted after the debate trigger has been evaluated. */
export interface TriggerEvaluatedEvent {
  type: "trigger_evaluated";
  result: TriggerResult;
}

/** Emitted after the spatial attention disagreement has been computed. */
export interface AttentionComputedEvent {
  type: "attention_computed";
  result: AttentionResult;
}

/** Emitted for each streamed token of an agent's argument. */
export interface ArgumentTokenEvent {
  type: "argument_token";
  agent: "A" | "B";
  token: string;
  round: 1 | 2;
}

/** Emitted when an agent has finished streaming its argument. */
export interface ArgumentDoneEvent {
  type: "argument_done";
  agent: "A" | "B";
  argument: string;
  round: 1 | 2;
}

/** Emitted when the final consensus has been reached. */
export interface ConsensusDoneEvent {
  type: "consensus_done";
  result: ConsensusResult;
}

/** Emitted when the pipeline fails irrecoverably. */
export interface ErrorEvent {
  type: "error";
  message: string;
}

/**
 * Discriminated union of every WebSocket event payload, keyed on the
 * literal `type` field. Mirrors the backend DebateEvent union exactly.
 */
export type DebateEvent =
  | AgentsRunningEvent
  | AgentsDoneEvent
  | TriggerEvaluatedEvent
  | AttentionComputedEvent
  | ArgumentTokenEvent
  | ArgumentDoneEvent
  | ConsensusDoneEvent
  | ErrorEvent;
