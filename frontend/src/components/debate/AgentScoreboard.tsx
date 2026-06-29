"use client";

/**
 * AgentScoreboard — a compact "scoreboard" for one agent in the debate. Unlike a
 * static result card, its probability bars and confidence update *live* every
 * turn as the agent revises its belief, so you can literally watch it change its
 * mind. A status line marks whether it is thinking, speaking, listening, or
 * settled, and the card glows in its accent while it holds the floor.
 */

import { AGENTS, getClassMeta, RISK_COLORS, type AgentId } from "@/lib/constants";
import { useCountup } from "@/hooks/useCountup";
import ProbabilityBars from "@/components/debate/ProbabilityBars";

export type AgentStatus = "thinking" | "speaking" | "listening" | "settled";

interface AgentScoreboardProps {
  agentId: AgentId;
  /** Live belief distribution (null until the agent has classified). */
  probs: Record<string, number> | null;
  confidence: number;
  topClass: string | null;
  status: AgentStatus;
}

const STATUS_LABEL: Record<AgentStatus, string> = {
  thinking: "Analyzing…",
  speaking: "Arguing",
  listening: "Listening",
  settled: "Settled",
};

export default function AgentScoreboard({
  agentId,
  probs,
  confidence,
  topClass,
  status,
}: AgentScoreboardProps): React.JSX.Element {
  const meta = AGENTS[agentId];
  const predMeta = topClass ? getClassMeta(topClass) : undefined;
  const conf = useCountup(confidence * 100, { duration: 600 });

  const speaking = status === "speaking";
  const glow = agentId === "A" ? "shadow-glow-a" : "shadow-glow-b";
  const breathe = agentId === "A" ? "animate-breathe-a" : "animate-breathe-b";

  return (
    <section
      aria-label={`${meta.label} — ${meta.name}`}
      data-agent={agentId}
      className={[
        "relative flex flex-col gap-3 rounded-2xl border bg-surface p-5 transition-[box-shadow,border-color] duration-300",
        speaking ? glow : "shadow-panel",
        status === "thinking" ? breathe : "",
      ].join(" ")}
      style={{ borderColor: speaking ? meta.color : "var(--border-subtle)" }}
    >
      <header className="flex items-start justify-between">
        <div>
          <div
            className="text-[11px] font-medium uppercase tracking-[0.12em]"
            style={{ color: meta.color }}
          >
            {meta.label}
          </div>
          <h3 className="font-display text-xl leading-tight text-ink">{meta.name}</h3>
          <div className="mt-0.5 text-[11px] uppercase tracking-wider text-ink-faint">
            {meta.descriptor}
          </div>
        </div>
        <span className="flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-ink-faint">
          {status === "settled" ? (
            <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" aria-hidden>
              <path d="M5 13l4 4L19 7" stroke={meta.color} strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          ) : (
            <span
              className={[
                "h-2 w-2 rounded-full",
                speaking ? "animate-pulse-dot" : "",
              ].join(" ")}
              style={{ backgroundColor: speaking ? meta.color : "var(--text-muted)" }}
            />
          )}
          {STATUS_LABEL[status]}
        </span>
      </header>

      {probs ? (
        <>
          <ProbabilityBars probabilities={probs} color={meta.color} predClass={topClass ?? undefined} max={4} />
          <footer className="flex items-end justify-between border-t border-hairline pt-3">
            <div>
              <div className="text-[10px] uppercase tracking-wider text-ink-faint">Current read</div>
              <div className="mt-0.5 flex items-center gap-2">
                <span className="font-display text-lg text-ink">
                  {predMeta?.fullName ?? topClass}
                </span>
                {topClass && (
                  <span
                    className="rounded-md border px-1.5 py-0.5 font-mono text-[10px] font-medium"
                    style={{
                      color: predMeta ? RISK_COLORS[predMeta.risk] : meta.color,
                      borderColor: predMeta ? RISK_COLORS[predMeta.risk] : meta.color,
                    }}
                  >
                    {topClass}
                  </span>
                )}
              </div>
            </div>
            <div className="text-right">
              <div className="text-[10px] uppercase tracking-wider text-ink-faint">Confidence</div>
              <div className="font-mono text-2xl font-medium tabular leading-none" style={{ color: meta.color }}>
                {conf.toFixed(0)}
                <span className="text-base text-ink-faint">%</span>
              </div>
            </div>
          </footer>
        </>
      ) : (
        <div className="flex items-center gap-2 py-6 text-xs text-ink-faint">
          <span className="flex gap-1">
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                className="h-1.5 w-1.5 rounded-full animate-dot-bounce"
                style={{ backgroundColor: meta.color, animationDelay: `${i * 0.15}s` }}
              />
            ))}
          </span>
          Analyzing the lesion…
        </div>
      )}
    </section>
  );
}
