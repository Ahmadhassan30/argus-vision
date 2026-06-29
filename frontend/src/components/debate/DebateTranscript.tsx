"use client";

/**
 * DebateTranscript — the heart of the experience: two agents arguing turn by
 * turn like people. Each turn arrives as a chat bubble (Agent A left/blue, Agent
 * B right/violet) that types itself out, tagged with the rhetorical move the
 * agent just made (opens, rebuts, softens, concedes, agrees). An agreement meter
 * climbs as their beliefs converge; when they finally agree, a closing chip
 * marks the resolution.
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
  open: "opens",
  press: "holds ground",
  rebut: "rebuts",
  soften: "softens",
  concede: "concedes",
  agree: "agrees",
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

  // Keep the latest turn in view as the conversation grows.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [turns.length]);

  const pct = Math.round(agreement * 100);
  const lastIndex = turns.length - 1;

  return (
    <section
      aria-label="Live debate transcript"
      className="rounded-2xl border border-hairline bg-surface p-6 shadow-panel animate-panel-enter"
    >
      {/* Header + agreement meter */}
      <header className="mb-4 flex items-center justify-between gap-4">
        <div>
          <div className="text-[11px] font-medium uppercase tracking-[0.12em] text-ink-faint">
            {finished ? "Debate concluded" : active ? "Live debate" : "Debate"}
          </div>
          <h3 className="font-display text-xl leading-tight text-ink">
            {round > 0 ? `Round ${round}` : "Opening statements"}
          </h3>
        </div>
        <div className="w-40 shrink-0">
          <div className="mb-1 flex items-center justify-between text-[10px] uppercase tracking-wider text-ink-faint">
            <span>Agreement</span>
            <span className="font-mono tabular text-ink-soft">{pct}%</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-surface-alt">
            <div
              className="h-full rounded-full transition-[width] duration-700"
              style={{
                width: `${pct}%`,
                background: "linear-gradient(90deg, #2563EB 0%, #7C3AED 50%, #059669 100%)",
              }}
            />
          </div>
        </div>
      </header>

      {/* Conversation */}
      <div ref={scrollRef} className="scroll-clinical flex max-h-[460px] flex-col gap-3 overflow-y-auto pr-1">
        {turns.length === 0 ? (
          <div className="flex items-center gap-2 py-10 text-sm text-ink-faint">
            <span className="flex gap-1">
              {[0, 1, 2].map((i) => (
                <span
                  key={i}
                  className="h-1.5 w-1.5 rounded-full bg-ink-faint animate-dot-bounce"
                  style={{ animationDelay: `${i * 0.15}s` }}
                />
              ))}
            </span>
            Opening statements incoming…
          </div>
        ) : (
          turns.map((turn) => {
            const meta = AGENTS[turn.agent];
            const isA = turn.agent === "A";
            const isLast = turn.index === lastIndex;
            return (
              <div key={turn.index} className={["flex", isA ? "justify-start" : "justify-end"].join(" ")}>
                <div
                  className="max-w-[86%] rounded-2xl border bg-surface-alt px-4 py-3"
                  style={{
                    borderColor: "var(--border-subtle)",
                    [isA ? "borderLeft" : "borderRight"]: `3px solid ${meta.color}`,
                  } as React.CSSProperties}
                >
                  <div className="mb-1 flex items-center gap-2">
                    <span className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: meta.color }}>
                      {meta.name}
                    </span>
                    <span className="rounded-full bg-surface px-1.5 py-px font-mono text-[9px] uppercase tracking-wider text-ink-faint">
                      {MOVE_LABEL[turn.move]}
                    </span>
                    <span className="font-mono text-[9px] text-ink-faint">R{turn.round}</span>
                  </div>
                  <p className="text-[13px] leading-relaxed text-ink-soft">
                    <ArgumentStream
                      text={turn.text}
                      agentId={turn.agent}
                      active={isLast && active}
                      speed={46}
                    />
                  </p>
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Resolution chip */}
      {finished && (
        <div className="mt-4 flex items-center justify-center">
          <span
            className="inline-flex items-center gap-2 rounded-full px-4 py-1.5 text-xs font-semibold"
            style={{
              backgroundColor: converged ? "rgba(5,150,105,0.12)" : "rgba(148,163,184,0.16)",
              color: converged ? "#059669" : "#475569",
            }}
          >
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: converged ? "#059669" : "#94A3B8" }} />
            {converged && convergedClass
              ? `Converged — agreed on ${getClassName(convergedClass)}`
              : "Debate concluded"}
          </span>
        </div>
      )}
    </section>
  );
}
