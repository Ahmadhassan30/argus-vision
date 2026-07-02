"use client";

/**
 * DebateTranscript — clean, clinical log-style debate view.
 *
 * Each turn is rendered inside a flat, dark workstation-style panel.
 * Uses `ArgumentStream` to show a human-paced typing animation with natural delays
 * and a blinking caret for the latest active turn.
 */

import { useEffect, useRef } from "react";

import { AGENTS, getClassName } from "@/lib/constants";
import type { DebateTurn } from "@/lib/debate/engine";
import type { Move } from "@/lib/debate/beliefs";
import ArgumentStream from "@/components/debate/ArgumentStream";

interface DebateTranscriptProps {
  turns: DebateTurn[];
  agreement: number;
  round: number;
  converged: boolean;
  finished: boolean;
  active: boolean;
  convergedClass?: string | null;
}

const MOVE_LABEL: Record<Move, string> = {
  open: "OPENS DEBATE",
  press: "REINFORCES READ",
  rebut: "REBUTS COUNTER",
  soften: "ADJUSTS BELIEF",
  concede: "CONCEDES READ",
  agree: "CONVERGES",
};

export default function DebateTranscript({
  turns,
  agreement,
  round,
  converged,
  finished,
  active,
  convergedClass,
}: DebateTranscriptProps): React.JSX.Element {
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const lastIndex = turns.length - 1;

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [turns.length]);

  const pct = Math.round(agreement * 100);

  return (
    <div className="p-5" style={{ backgroundColor: "#0a0a0c" }}>
      {/* Header Bar */}
      <div className="flex items-center justify-between border-b pb-3 mb-4" style={{ borderColor: "#1a1a1f" }}>
        <div className="flex items-center gap-3">
          <span className="font-mono text-[10px] uppercase tracking-[0.2em]" style={{ color: "#8e9196" }}>
            TRANSACTION LOG
          </span>
          <span className="h-1 w-1 rounded-full bg-ink-faint" />
          <span className="font-mono text-[10px] uppercase text-[#6b7280]">
            {finished ? "Concluded" : active ? "Active Stream" : "Standby"}
          </span>
        </div>

        {/* Agreement Meter */}
        <div className="flex items-center gap-3">
          <span className="font-mono text-[10px] uppercase tracking-wider text-[#6b7280]">
            Consensus Agreement
          </span>
          <div className="h-1 w-20 overflow-hidden rounded-full bg-[#1a1a1f]">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${pct}%`,
                backgroundColor: pct > 80 ? "#059669" : "#3b82f6",
              }}
            />
          </div>
          <span className="font-mono text-[10px] font-semibold tabular text-[#e5e7eb]">
            {pct}%
          </span>
        </div>
      </div>

      {/* Log Feed */}
      <div
        ref={scrollRef}
        className="flex max-h-[380px] flex-col gap-3 overflow-y-auto pr-1 scroll-clinical"
      >
        {turns.length === 0 ? (
          <div className="flex items-center gap-2 py-10 justify-center font-mono text-[11px]" style={{ color: "#4b5563" }}>
            <span className="h-1.5 w-1.5 rounded-full animate-ping" style={{ backgroundColor: "#3b82f6" }} />
            INITIALIZING LOG BUFFER...
          </div>
        ) : (
          turns.map((turn) => {
            const meta = AGENTS[turn.agent];
            const isLast = turn.index === lastIndex;
            return (
              <div
                key={turn.index}
                className="rounded border p-3 flex flex-col gap-2 transition-all duration-300"
                style={{
                  backgroundColor: "#0d0d0f",
                  borderColor: isLast && active ? meta.color : "#1a1a1f",
                  boxShadow: isLast && active ? `inset 2px 0 0 ${meta.color}` : "none",
                }}
              >
                {/* Meta details */}
                <div className="flex items-center justify-between border-b pb-1.5" style={{ borderColor: "#141417" }}>
                  <div className="flex items-center gap-2">
                    <span
                      className="font-mono text-[9px] font-bold uppercase tracking-widest"
                      style={{ color: meta.color }}
                    >
                      {meta.name}
                    </span>
                    <span className="font-mono text-[9px]" style={{ color: "#4b5563" }}>
                      [ROUND {turn.round}]
                    </span>
                  </div>

                  <span
                    className="font-mono text-[8px] font-semibold tracking-wider rounded px-1"
                    style={{
                      backgroundColor: `${meta.color}15`,
                      color: meta.color,
                      border: `1px solid ${meta.color}30`,
                    }}
                  >
                    {MOVE_LABEL[turn.move]}
                  </span>
                </div>

                {/* Body Text */}
                <div style={{ color: isLast && active ? "#e5e7eb" : "#a1a1a6" }}>
                  <ArgumentStream
                    text={turn.text}
                    agentId={turn.agent}
                    active={isLast && active}
                  />
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Audit Verdict Banner */}
      {finished && (
        <div
          className="mt-4 flex items-center justify-between rounded border p-3 font-mono text-[10px] tracking-wide"
          style={{
            backgroundColor: converged ? "#081c15" : "#141417",
            borderColor: converged ? "#0f3d2a" : "#1f1f23",
            color: converged ? "#34d399" : "#9ca3af",
          }}
        >
          <span>LOG STATUS: VERDICT LOCKED</span>
          <span>
            {converged && convergedClass
              ? `AGREEMENT: ${getClassName(convergedClass)}`
              : "TERMINATED — NO CONVERGENCE"}
          </span>
        </div>
      )}
    </div>
  );
}
