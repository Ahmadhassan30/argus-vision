/**
 * Turn-taking debate engine.
 *
 * Drives the alternating conversation: Agent A opens, Agent B listens and
 * answers, A rethinks and revises, and so on — each turn a real belief revision
 * (see beliefs.ts) plus a retrieved, opponent-referencing line (see retrieval.ts).
 * The loop runs until the two beliefs converge (low JS divergence), then exchanges
 * a brief closing handshake, with a hard safety cap so it always terminates.
 *
 * The engine is pure-ish: `seedDebate` makes the initial state from the real
 * classifier outputs, and `nextTurn` produces the next turn given the current
 * dynamics (spatial evidence and, once known, the calibrated consensus). The
 * React hook owns timing.
 */

import {
  jsDivergence,
  normalize,
  revise,
  top,
  toRecord,
  type Move,
  type Vec,
} from "@/lib/debate/beliefs";
import { retrieveTurn, type TurnContext } from "@/lib/debate/retrieval";

/** Convergence threshold on JS divergence (bits). */
export const CONVERGE_AT = 0.03;
/** Hard safety cap on total turns. */
export const MAX_TURNS = 18;

/** One spoken turn in the transcript. */
export interface DebateTurn {
  index: number;
  round: number;
  agent: "A" | "B";
  text: string;
  move: Move;
  topClass: string;
  confidence: number;
  /** JS divergence to the opponent after this turn. */
  js: number;
}

/** Mutable engine state, held in a ref by the hook. */
export interface EngineState {
  beliefA: Vec;
  beliefB: Vec;
  /** Original classifier leading classes (anchor the spatial evidence). */
  topA0: string;
  topB0: string;
  speaker: "A" | "B";
  turnIndex: number;
  recentIds: string[];
  converged: boolean;
  /** Closing handshake counter: -1 = not closing, 0 = final closing turn next. */
  agreeLeft: number;
  finished: boolean;
  /** Initial JS divergence, for the agreement meter. */
  jsRef: number;
}

/** Dynamics that may change mid-debate as the pipeline progresses. */
export interface Dynamics {
  /** Spatial-evidence log-bias (once attention is computed), else null. */
  evidence: Vec | null;
  /** Consensus attractor distribution (once known), else null. */
  consensus: Vec | null;
}

export interface StepResult {
  state: EngineState;
  turn: DebateTurn;
}

/** Seed the debate from the two agents' real probability vectors. */
export function seedDebate(
  beliefA: Vec,
  beliefB: Vec,
  topA0: string,
  topB0: string,
): EngineState {
  return {
    beliefA,
    beliefB,
    topA0,
    topB0,
    speaker: "A",
    turnIndex: 0,
    recentIds: [],
    converged: false,
    agreeLeft: -1,
    finished: false,
    jsRef: Math.max(jsDivergence(beliefA, beliefB), 0.05),
  };
}

/** Blend `self` toward `opp` (used for the closing handshake alignment). */
function blendToward(self: Vec, opp: Vec, k: number): Vec {
  return normalize(self.map((x, i) => x * (1 - k) + opp[i] * k));
}

/** Produce the next turn and the resulting engine state. */
export async function nextTurn(s: EngineState, dyn: Dynamics): Promise<StepResult> {
  const speaker = s.speaker;
  const self = speaker === "A" ? s.beliefA : s.beliefB;
  const opp = speaker === "A" ? s.beliefB : s.beliefA;
  const opening = s.turnIndex < 2;
  const closing = s.agreeLeft >= 0;

  let move: Move;
  let nextSelf: Vec;

  if (opening) {
    move = "open";
    nextSelf = self;
  } else if (closing) {
    move = "agree";
    nextSelf = blendToward(self, opp, 0.6);
  } else {
    const r = revise({
      self,
      opp,
      evidence: dyn.evidence,
      consensus: dyn.consensus,
      opening: false,
      convergeAt: CONVERGE_AT,
    });
    move = r.move;
    nextSelf = r.next;
  }

  const selfTop = top(nextSelf);
  const oppTop = top(opp);

  const ctx: TurnContext = {
    agent: speaker,
    move,
    selfClass: selfTop.cls,
    oppClass: oppTop.cls,
    recentIds: s.recentIds,
  };
  const retrieved = await retrieveTurn(ctx);

  const beliefA = speaker === "A" ? nextSelf : s.beliefA;
  const beliefB = speaker === "B" ? nextSelf : s.beliefB;
  const js = jsDivergence(beliefA, beliefB);

  const turn: DebateTurn = {
    index: s.turnIndex,
    round: Math.floor(s.turnIndex / 2) + 1,
    agent: speaker,
    text: retrieved.text,
    move,
    topClass: selfTop.cls,
    confidence: selfTop.p,
    js,
  };

  // Termination bookkeeping.
  let agreeLeft = s.agreeLeft;
  let finished = false;
  if (closing) {
    agreeLeft = s.agreeLeft - 1;
    if (agreeLeft < 0) finished = true;
  } else if (!opening && js <= CONVERGE_AT) {
    agreeLeft = 0; // one closing turn from the other agent, then finish
  }
  if (!finished && s.turnIndex + 1 >= MAX_TURNS) finished = true;

  const state: EngineState = {
    ...s,
    beliefA,
    beliefB,
    speaker: speaker === "A" ? "B" : "A",
    turnIndex: s.turnIndex + 1,
    recentIds: [...s.recentIds, retrieved.id].slice(-10),
    converged: js <= CONVERGE_AT,
    agreeLeft,
    finished,
  };

  return { state, turn };
}

/** Agreement level in [0,1] for the meter: 0 = initial gap, 1 = converged. */
export function agreementOf(state: EngineState): number {
  const js = jsDivergence(state.beliefA, state.beliefB);
  return Math.max(0, Math.min(1, 1 - js / state.jsRef));
}

/** Convenience: current beliefs as class→prob maps. */
export function beliefsAsRecords(state: EngineState): {
  A: Record<string, number>;
  B: Record<string, number>;
} {
  return { A: toRecord(state.beliefA), B: toRecord(state.beliefB) };
}
