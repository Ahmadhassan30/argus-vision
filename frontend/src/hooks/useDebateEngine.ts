"use client";

/**
 * useDebateEngine — drives the turn-taking debate on top of the live classifier
 * stream. It seeds two beliefs from the agents' real distributions, then paces
 * the alternating conversation (A → B → A …) one turn at a time, folding in the
 * spatial evidence and the calibrated consensus as they arrive, until the two
 * beliefs converge. Each turn's display time scales with its length so it reads
 * like genuine deliberation; the whole exchange runs ~1–2 minutes and ends when
 * the agents actually agree (or a safety cap is hit).
 */

import { useEffect, useRef, useState } from "react";

import type { DebateState } from "@/lib/debateReducer";
import { evidenceBias, top, toRecord, toVec } from "@/lib/debate/beliefs";
import {
  agreementOf,
  nextTurn,
  seedDebate,
  type DebateTurn,
  type Dynamics,
  type EngineState,
} from "@/lib/debate/engine";

export interface DebateEngineView {
  turns: DebateTurn[];
  beliefA: Record<string, number>;
  beliefB: Record<string, number>;
  confA: number;
  confB: number;
  /** Who is speaking the latest turn, or null when idle/finished. */
  speaker: "A" | "B" | null;
  round: number;
  /** Agreement level 0→1 (1 = converged). */
  agreement: number;
  js: number;
  converged: boolean;
  finished: boolean;
  /** The debate has been seeded and is producing turns. */
  active: boolean;
}

function initialView(): DebateEngineView {
  return {
    turns: [],
    beliefA: {},
    beliefB: {},
    confA: 0,
    confB: 0,
    speaker: null,
    round: 0,
    agreement: 0,
    js: 1,
    converged: false,
    finished: false,
    active: false,
  };
}

/** Display time for a turn — long enough to type it out and read it. */
function delayFor(turn: DebateTurn): number {
  return Math.max(4200, Math.min(11000, turn.text.length * 55 + 1600));
}

export function useDebateEngine(ws: DebateState, jobId: string): DebateEngineView {
  const [view, setView] = useState<DebateEngineView>(initialView);

  const wsRef = useRef(ws);
  wsRef.current = ws;

  const engineRef = useRef<EngineState | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const cancelledRef = useRef(false);
  const startedRef = useRef(false);

  // Reset on job change.
  useEffect(() => {
    cancelledRef.current = false;
    startedRef.current = false;
    engineRef.current = null;
    setView(initialView());
    return () => {
      cancelledRef.current = true;
      if (timerRef.current !== null) clearTimeout(timerRef.current);
    };
  }, [jobId]);

  // Seed and run once both agents have classified.
  useEffect(() => {
    if (startedRef.current) return;
    const a = ws.agentA;
    const b = ws.agentB;
    if (!a || !b) return;
    startedRef.current = true;

    engineRef.current = seedDebate(
      toVec(a.result.probabilities),
      toVec(b.result.probabilities),
      a.result.pred_class,
      b.result.pred_class,
    );

    const buildDynamics = (): Dynamics => {
      const s = wsRef.current;
      const eng = engineRef.current;
      const evidence =
        s.attention && eng
          ? evidenceBias(
              s.attention.region_stats_a,
              s.attention.region_stats_b,
              eng.topA0,
              eng.topB0,
            )
          : null;
      const consensus = s.consensus ? toVec(s.consensus.probabilities) : null;
      return { evidence, consensus };
    };

    const tick = async (): Promise<void> => {
      if (cancelledRef.current) return;
      const eng = engineRef.current;
      if (!eng || eng.finished) return;

      const { state, turn } = await nextTurn(eng, buildDynamics());
      if (cancelledRef.current) return;
      engineRef.current = state;

      setView((v) => ({
        turns: [...v.turns, turn],
        beliefA: toRecord(state.beliefA),
        beliefB: toRecord(state.beliefB),
        confA: top(state.beliefA).p,
        confB: top(state.beliefB).p,
        speaker: state.finished ? null : state.speaker,
        round: turn.round,
        agreement: agreementOf(state),
        js: turn.js,
        converged: state.converged,
        finished: state.finished,
        active: !state.finished,
      }));

      if (!state.finished) {
        timerRef.current = setTimeout(() => void tick(), delayFor(turn));
      }
    };

    setView((v) => ({ ...v, active: true, speaker: "A" }));
    timerRef.current = setTimeout(() => void tick(), 900);
  }, [ws.agentA, ws.agentB, jobId]);

  return view;
}
