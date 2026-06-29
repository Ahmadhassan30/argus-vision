/**
 * Pure reducer for the Argus Vision debate stream.
 *
 * The backend publishes a discriminated union of {@link DebateEvent} payloads
 * over the WebSocket, and — crucially — replays a full {@link JobResult}
 * snapshot the moment a client connects so late subscribers can catch up. This
 * reducer folds both shapes into one flat, render-ready {@link DebateState},
 * stamping each pipeline milestone with the time it completed so the timeline
 * rail can show real durations.
 *
 * Argument *text* is no longer produced by the backend (the former Groq debate
 * generation was removed); it is generated client-side and lives in a separate
 * hook. This reducer therefore concerns itself only with the numerical pipeline.
 */

import type {
  AgentResult,
  AttentionResult,
  ConsensusResult,
  DebateEvent,
  JobResult,
  JobStatus,
  TriggerResult,
} from "@/types/debate";

/** Identifier for one milestone in the vertical timeline rail. */
export type StepKey =
  | "uploaded"
  | "agents_init"
  | "analysis_a"
  | "analysis_b"
  | "divergence"
  | "path"
  | "attention"
  | "consensus"
  | "delivered";

/** Coarse UI phase driving the header status pill and ambient background. */
export type UiPhase =
  | "connecting"
  | "running"
  | "agents_done"
  | "triggered"
  | "fast_path"
  | "attention"
  | "consensus"
  | "resolved"
  | "error";

/** Flattened, render-ready projection of every event observed so far. */
export interface DebateState {
  status: JobStatus;
  phase: UiPhase;
  agentA: AgentResult | null;
  agentB: AgentResult | null;
  trigger: TriggerResult | null;
  attention: AttentionResult | null;
  consensus: ConsensusResult | null;
  error: string | null;
  isConnected: boolean;
  /** Whether the divergence trigger fired (true) or the fast path was taken. */
  triggerFired: boolean | null;
  /** Epoch-ms completion time for each reached milestone. */
  completedAt: Partial<Record<StepKey, number>>;
}

/** Actions the hook dispatches into the reducer. */
export type DebateAction =
  | { kind: "connected"; at: number }
  | { kind: "disconnected" }
  | { kind: "snapshot"; job: JobResult; at: number }
  | { kind: "event"; event: DebateEvent; at: number }
  | { kind: "reset" };

/** The pristine state used on mount and whenever the job id changes. */
export function createInitialState(): DebateState {
  return {
    status: "queued",
    phase: "connecting",
    agentA: null,
    agentB: null,
    trigger: null,
    attention: null,
    consensus: null,
    error: null,
    isConnected: false,
    triggerFired: null,
    completedAt: {},
  };
}

/** Set `key` to `at` only if it has not already been stamped (first-wins). */
function stamp(
  map: Partial<Record<StepKey, number>>,
  at: number,
  ...keys: StepKey[]
): Partial<Record<StepKey, number>> {
  let next = map;
  for (const key of keys) {
    if (next[key] === undefined) {
      next = { ...next, [key]: at };
    }
  }
  return next;
}

/** Derive the coarse UI phase from the accumulated state. */
function derivePhase(s: DebateState): UiPhase {
  if (s.error !== null) return "error";
  if (s.consensus !== null) return "resolved";
  if (s.trigger !== null) {
    if (s.trigger.fired) {
      return s.attention === null ? "attention" : "consensus";
    }
    return "fast_path";
  }
  if (s.agentA !== null && s.agentB !== null) return "agents_done";
  if (s.status === "running" || s.status === "queued") {
    return s.isConnected ? "running" : "connecting";
  }
  return s.isConnected ? "running" : "connecting";
}

/** Recompute derived fields (phase) after a structural change. */
function recompute(s: DebateState): DebateState {
  const phase = derivePhase(s);
  return phase === s.phase ? s : { ...s, phase };
}

/** Fold a full job snapshot into state (used on connect / reconnect). */
function applySnapshot(prev: DebateState, job: JobResult, at: number): DebateState {
  let completedAt: Partial<Record<StepKey, number>> = { ...prev.completedAt };
  completedAt = stamp(completedAt, at, "uploaded");
  if (job.agent_a || job.agent_b) {
    completedAt = stamp(completedAt, at, "agents_init", "analysis_a", "analysis_b");
  }
  if (job.trigger) completedAt = stamp(completedAt, at, "divergence", "path");
  if (job.attention) completedAt = stamp(completedAt, at, "attention");
  if (job.consensus) completedAt = stamp(completedAt, at, "consensus", "delivered");

  return recompute({
    ...prev,
    status: job.status,
    agentA: job.agent_a ?? prev.agentA,
    agentB: job.agent_b ?? prev.agentB,
    trigger: job.trigger ?? prev.trigger,
    attention: job.attention ?? prev.attention,
    consensus: job.consensus ?? prev.consensus,
    error: job.error ?? prev.error,
    triggerFired: job.trigger ? job.trigger.fired : prev.triggerFired,
    completedAt,
  });
}

/** Fold a single live event into state. */
function applyEvent(prev: DebateState, event: DebateEvent, at: number): DebateState {
  switch (event.type) {
    case "agents_running":
      return recompute({
        ...prev,
        status: "running",
        completedAt: stamp(prev.completedAt, at, "uploaded", "agents_init"),
      });

    case "agents_done":
      return recompute({
        ...prev,
        agentA: event.agent_a,
        agentB: event.agent_b,
        // Keep the most advanced status; the second agents_done (with heatmaps)
        // arrives after the trigger, so never regress the status here.
        status:
          prev.status === "queued" || prev.status === "running"
            ? "agents_done"
            : prev.status,
        completedAt: stamp(
          prev.completedAt,
          at,
          "uploaded",
          "agents_init",
          "analysis_a",
          "analysis_b",
        ),
      });

    case "trigger_evaluated":
      return recompute({
        ...prev,
        trigger: event.result,
        triggerFired: event.result.fired,
        status: "trigger_evaluated",
        completedAt: stamp(prev.completedAt, at, "divergence", "path"),
      });

    case "attention_computed":
      return recompute({
        ...prev,
        attention: event.result,
        status: "attention_computed",
        completedAt: stamp(prev.completedAt, at, "attention"),
      });

    case "consensus_done":
      return recompute({
        ...prev,
        consensus: event.result,
        status: "consensus_done",
        completedAt: stamp(prev.completedAt, at, "consensus", "delivered"),
      });

    case "error":
      return recompute({ ...prev, error: event.message, status: "failed" });

    // The backend no longer emits argument tokens; ignore for forward-compat.
    case "argument_token":
    case "argument_done":
    default:
      return prev;
  }
}

/** The reducer. */
export function debateReducer(prev: DebateState, action: DebateAction): DebateState {
  switch (action.kind) {
    case "connected":
      return recompute({
        ...prev,
        isConnected: true,
        completedAt: stamp(prev.completedAt, action.at, "uploaded"),
      });
    case "disconnected":
      return prev.isConnected ? recompute({ ...prev, isConnected: false }) : prev;
    case "snapshot":
      return applySnapshot(prev, action.job, action.at);
    case "event":
      return applyEvent(prev, action.event, action.at);
    case "reset":
      return createInitialState();
    default:
      return prev;
  }
}

/** Whether a job status means the pipeline has terminated (no reconnects). */
export function isTerminalStatus(status: JobStatus): boolean {
  return status === "consensus_done" || status === "failed";
}

/** Type guard distinguishing a full {@link JobResult} snapshot from an event. */
export function isJobSnapshot(value: unknown): value is JobResult {
  return (
    typeof value === "object" &&
    value !== null &&
    "job_id" in value &&
    "status" in value &&
    !("type" in value)
  );
}
