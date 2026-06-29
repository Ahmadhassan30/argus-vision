"use client";

/**
 * AgentCard — one mind in the arena. Shows the agent's identity, a live argument
 * stream (its reasoning, then its rebuttal), the full probability distribution,
 * and its verdict. It "thinks" (animated dots) until its result lands, glows in
 * its accent colour while it is actively arguing, and never shifts layout thanks
 * to a fixed-height argument well.
 */

import type { AgentResult } from "@/types/debate";
import { AGENTS, getClassMeta, RISK_COLORS, type AgentId } from "@/lib/constants";
import { useCountup } from "@/hooks/useCountup";
import ArgumentStream from "@/components/debate/ArgumentStream";
import ProbabilityBars from "@/components/debate/ProbabilityBars";

interface AgentCardProps {
  agentId: AgentId;
  result: AgentResult | null;
  round1: string;
  rebuttal: string;
  /** This agent is currently streaming an argument. */
  argumentActive: boolean;
  /** Agents are running; no result yet. */
  thinking: boolean;
  className?: string;
}

function ThinkingDots({ label, color }: { label: string; color: string }): React.JSX.Element {
  return (
    <div className="flex items-center gap-2 text-xs text-ink-faint">
      <span className="flex gap-1">
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className="h-1.5 w-1.5 rounded-full animate-dot-bounce"
            style={{ backgroundColor: color, animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </span>
      <span>{label}</span>
    </div>
  );
}

export default function AgentCard({
  agentId,
  result,
  round1,
  rebuttal,
  argumentActive,
  thinking,
  className = "",
}: AgentCardProps): React.JSX.Element {
  const meta = AGENTS[agentId];
  const classification = result?.result ?? null;
  const predMeta = classification ? getClassMeta(classification.pred_class) : undefined;
  const conf = useCountup(classification ? classification.confidence * 100 : 0, {
    duration: 1100,
  });

  const active = argumentActive;
  const glowClass = agentId === "A" ? "shadow-glow-a" : "shadow-glow-b";
  const breatheClass = agentId === "A" ? "animate-breathe-a" : "animate-breathe-b";

  return (
    <section
      aria-label={`${meta.label} — ${meta.name}`}
      data-agent={agentId}
      className={[
        "relative flex flex-col gap-4 rounded-2xl border bg-surface p-6 transition-[box-shadow,border-color] duration-300",
        active ? glowClass : "shadow-panel",
        thinking && !active ? breatheClass : "",
        className,
      ].join(" ")}
      style={{ borderColor: active ? meta.color : "var(--border-subtle)" }}
    >
      {/* Identity */}
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
        {active && (
          <span className="flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-wider text-ink-faint">
            <span
              className="h-2 w-2 rounded-full animate-pulse-dot"
              style={{ backgroundColor: meta.color }}
            />
            arguing
          </span>
        )}
      </header>

      {/* Argument well — fixed height so panels never jump. */}
      <div
        className="scroll-clinical max-h-[220px] min-h-[128px] overflow-y-auto rounded-r-lg py-3 pl-4 pr-3 text-[13px] leading-[1.7] text-ink-soft"
        style={{
          backgroundColor: "var(--bg-surface-alt)",
          borderLeft: `3px solid ${meta.color}`,
        }}
      >
        {!classification ? (
          <ThinkingDots label="Analyzing the lesion…" color={meta.color} />
        ) : round1.length === 0 ? (
          <ThinkingDots label="Formulating argument…" color={meta.color} />
        ) : (
          <>
            <ArgumentStream
              text={round1}
              agentId={agentId}
              active={argumentActive && rebuttal.length === 0}
            />
            {rebuttal.length > 0 && (
              <div className="mt-3 border-t border-hairline pt-3">
                <div
                  className="mb-1 text-[10px] font-semibold uppercase tracking-wider"
                  style={{ color: meta.color }}
                >
                  ↩ Rebuttal
                </div>
                <ArgumentStream text={rebuttal} agentId={agentId} active={argumentActive} />
              </div>
            )}
          </>
        )}
      </div>

      {/* Distribution */}
      {classification && (
        <ProbabilityBars
          probabilities={classification.probabilities}
          color={meta.color}
          predClass={classification.pred_class}
        />
      )}

      {/* Verdict */}
      {classification && (
        <footer className="flex items-end justify-between border-t border-hairline pt-4">
          <div>
            <div className="text-[10px] uppercase tracking-wider text-ink-faint">
              Leading read
            </div>
            <div className="mt-0.5 flex items-center gap-2">
              <span className="font-display text-lg text-ink">
                {predMeta?.fullName ?? classification.pred_class}
              </span>
              <span
                className="rounded-md border px-1.5 py-0.5 font-mono text-[10px] font-medium"
                style={{
                  color: predMeta ? RISK_COLORS[predMeta.risk] : meta.color,
                  borderColor: predMeta ? RISK_COLORS[predMeta.risk] : meta.color,
                }}
              >
                {classification.pred_class}
              </span>
            </div>
          </div>
          <div className="text-right">
            <div className="text-[10px] uppercase tracking-wider text-ink-faint">
              Confidence
            </div>
            <div
              className="font-mono text-2xl font-medium tabular leading-none"
              style={{ color: meta.color }}
            >
              {conf.toFixed(0)}
              <span className="text-base text-ink-faint">%</span>
            </div>
          </div>
        </footer>
      )}
    </section>
  );
}
