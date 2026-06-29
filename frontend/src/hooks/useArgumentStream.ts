"use client";

/**
 * useArgumentStream — orchestrates the live debate arguments on top of the
 * numerical {@link DebateState}. Because the backend no longer generates debate
 * text, this hook synthesises it client-side via the free-LLM proxy (with a
 * deterministic fallback), firing each phase at the dramatically right moment:
 *
 *   • round 1  — the instant both agents have classified;
 *   • rebuttal — a beat after the divergence trigger FIRES (so the user can read
 *                the openers first);
 *   • synthesis — just after the consensus resolves.
 *
 * It is fully non-blocking: failures degrade to fallback text and never affect
 * the rest of the UI. All in-flight requests abort on unmount or job change.
 */

import { useEffect, useRef, useState } from "react";

import type { DebateState } from "@/lib/debateReducer";
import {
  rebuttalMessages,
  round1Messages,
  synthesisMessages,
} from "@/lib/llm/prompts";
import {
  templateRebuttal,
  templateRound1,
  templateSynthesis,
} from "@/lib/llm/argumentTemplates";
import { streamArgument } from "@/lib/llm/streamArgument";

/** Delay (ms) after the trigger fires before rebuttals begin. */
const REBUTTAL_DELAY_MS = 2200;
/** Delay (ms) after consensus before the synthesis begins. */
const SYNTHESIS_DELAY_MS = 600;

interface Pair {
  A: string;
  B: string;
}

interface StreamingFlags {
  round1A: boolean;
  round1B: boolean;
  rebuttalA: boolean;
  rebuttalB: boolean;
  synthesis: boolean;
}

export interface ArgumentStreamState {
  round1: Pair;
  rebuttal: Pair;
  synthesis: string;
  streaming: StreamingFlags;
  /** Convenience: is either argument for this agent currently streaming. */
  agentActive: { A: boolean; B: boolean };
}

const EMPTY: ArgumentStreamState = {
  round1: { A: "", B: "" },
  rebuttal: { A: "", B: "" },
  synthesis: "",
  streaming: {
    round1A: false,
    round1B: false,
    rebuttalA: false,
    rebuttalB: false,
    synthesis: false,
  },
  agentActive: { A: false, B: false },
};

export function useArgumentStream(
  state: DebateState,
  jobId: string,
): ArgumentStreamState {
  const [round1, setRound1] = useState<Pair>({ A: "", B: "" });
  const [rebuttal, setRebuttal] = useState<Pair>({ A: "", B: "" });
  const [synthesis, setSynthesis] = useState("");
  const [streaming, setStreaming] = useState<StreamingFlags>(EMPTY.streaming);

  const startedRound1 = useRef(false);
  const startedRebuttal = useRef(false);
  const startedSynthesis = useRef(false);
  const controllers = useRef<AbortController[]>([]);
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]);

  // Reset everything when the job changes.
  useEffect(() => {
    startedRound1.current = false;
    startedRebuttal.current = false;
    startedSynthesis.current = false;
    setRound1({ A: "", B: "" });
    setRebuttal({ A: "", B: "" });
    setSynthesis("");
    setStreaming(EMPTY.streaming);

    const acs = controllers.current;
    const tms = timers.current;
    return () => {
      for (const ac of acs) ac.abort();
      for (const t of tms) clearTimeout(t);
      controllers.current = [];
      timers.current = [];
    };
  }, [jobId]);

  // Round 1 — fire the moment both agents have classified.
  useEffect(() => {
    if (startedRound1.current) return;
    const a = state.agentA;
    const b = state.agentB;
    if (!a || !b) return;
    startedRound1.current = true;

    const acA = new AbortController();
    const acB = new AbortController();
    controllers.current.push(acA, acB);

    setStreaming((s) => ({ ...s, round1A: true, round1B: true }));

    void streamArgument({
      messages: round1Messages("A", a.result),
      fallback: templateRound1("A", a.result),
      signal: acA.signal,
      onText: (t) => setRound1((p) => ({ ...p, A: t })),
    }).finally(() => setStreaming((s) => ({ ...s, round1A: false })));

    void streamArgument({
      messages: round1Messages("B", b.result),
      fallback: templateRound1("B", b.result),
      signal: acB.signal,
      onText: (t) => setRound1((p) => ({ ...p, B: t })),
    }).finally(() => setStreaming((s) => ({ ...s, round1B: false })));
  }, [state.agentA, state.agentB]);

  // Rebuttal — a beat after the trigger fires.
  useEffect(() => {
    if (startedRebuttal.current) return;
    if (state.triggerFired !== true) return;
    const a = state.agentA;
    const b = state.agentB;
    if (!a || !b) return;
    startedRebuttal.current = true;

    const timer = setTimeout(() => {
      const acA = new AbortController();
      const acB = new AbortController();
      controllers.current.push(acA, acB);
      setStreaming((s) => ({ ...s, rebuttalA: true, rebuttalB: true }));

      void streamArgument({
        messages: rebuttalMessages("A", a.result, b.result),
        fallback: templateRebuttal("A", a.result, b.result),
        signal: acA.signal,
        onText: (t) => setRebuttal((p) => ({ ...p, A: t })),
      }).finally(() => setStreaming((s) => ({ ...s, rebuttalA: false })));

      void streamArgument({
        messages: rebuttalMessages("B", b.result, a.result),
        fallback: templateRebuttal("B", b.result, a.result),
        signal: acB.signal,
        onText: (t) => setRebuttal((p) => ({ ...p, B: t })),
      }).finally(() => setStreaming((s) => ({ ...s, rebuttalB: false })));
    }, REBUTTAL_DELAY_MS);

    timers.current.push(timer);
  }, [state.triggerFired, state.agentA, state.agentB]);

  // Synthesis — just after consensus resolves.
  useEffect(() => {
    if (startedSynthesis.current) return;
    const a = state.agentA;
    const b = state.agentB;
    const consensus = state.consensus;
    if (!a || !b || !consensus) return;
    startedSynthesis.current = true;

    const timer = setTimeout(() => {
      const ac = new AbortController();
      controllers.current.push(ac);
      setStreaming((s) => ({ ...s, synthesis: true }));

      void streamArgument({
        messages: synthesisMessages(a.result, b.result, state.trigger, consensus),
        fallback: templateSynthesis(a.result, b.result, state.trigger, consensus),
        signal: ac.signal,
        onText: (t) => setSynthesis(t),
      }).finally(() => setStreaming((s) => ({ ...s, synthesis: false })));
    }, SYNTHESIS_DELAY_MS);

    timers.current.push(timer);
  }, [state.consensus, state.agentA, state.agentB, state.trigger]);

  return {
    round1,
    rebuttal,
    synthesis,
    streaming,
    agentActive: {
      A: streaming.round1A || streaming.rebuttalA,
      B: streaming.round1B || streaming.rebuttalB,
    },
  };
}
