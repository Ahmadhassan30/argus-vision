"use client";

/**
 * AgentScoreboard — compact clinical data card for one agent.
 *
 * Flat, dark panel with no glow/breathe animations. Shows the agent's
 * identity, top prediction, confidence, and probability bars. Data-first,
 * no decorative ornamentation.
 */

import { AGENTS, getClassMeta, RISK_COLORS, type AgentId } from "@/lib/constants";
import { useCountup } from "@/hooks/useCountup";
import ProbabilityBars from "@/components/debate/ProbabilityBars";

export type AgentStatus = "thinking" | "speaking" | "listening" | "settled";

interface AgentScoreboardProps {
  agentId: AgentId;
  probs: Record<string, number> | null;
  confidence: number;
  topClass: string | null;
  status: AgentStatus;
}

const STATUS_LABEL: Record<AgentStatus, string> = {
  thinking: "Analyzing",
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

  return (
    <div className="p-5">
      {/* Header row: agent identity + status */}
      <div className="flex items-start justify-between">
        <div>
          <div
            className="font-mono text-[10px] font-medium uppercase tracking-widest"
            style={{ color: meta.color }}
          >
            {meta.label}
          </div>
          <div className="mt-0.5 text-sm font-medium" style={{ color: "#e5e7eb" }}>
            {meta.name}
          </div>
          <div className="font-mono text-[10px] uppercase tracking-wider" style={{ color: "#4b5563" }}>
            {meta.descriptor}
          </div>
        </div>
        <span className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-wider" style={{ color: "#6b7280" }}>
          {status === "settled" ? (
            <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" aria-hidden>
              <path d="M5 13l4 4L19 7" stroke={meta.color} strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          ) : (
            <span
              className="h-1.5 w-1.5 rounded-full"
              style={{ backgroundColor: status === "speaking" ? meta.color : "#4b5563" }}
            />
          )}
          {STATUS_LABEL[status]}
        </span>
      </div>

      {probs ? (
        <>
          {/* Probability bars */}
          <div className="mt-4">
            <ProbabilityBars probabilities={probs} color={meta.color} predClass={topClass ?? undefined} max={4} />
          </div>

          {/* Bottom row: prediction + confidence */}
          <div
            className="mt-4 flex items-end justify-between border-t pt-3"
            style={{ borderColor: "#1a1a1f" }}
          >
            <div>
              <div className="font-mono text-[10px] uppercase tracking-wider" style={{ color: "#6b7280" }}>
                Prediction
              </div>
              <div className="mt-0.5 flex items-center gap-2">
                <span className="text-base font-medium" style={{ color: "#e5e7eb" }}>
                  {predMeta?.fullName ?? topClass}
                </span>
                {topClass && (
                  <span
                    className="rounded border px-1.5 py-0.5 font-mono text-[10px] font-medium"
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
              <div className="font-mono text-[10px] uppercase tracking-wider" style={{ color: "#6b7280" }}>
                Confidence
              </div>
              <div className="font-mono text-xl font-medium tabular leading-none" style={{ color: meta.color }}>
                {conf.toFixed(0)}
                <span className="text-sm" style={{ color: "#6b7280" }}>%</span>
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="mt-6 flex items-center gap-2 py-4 font-mono text-[11px]" style={{ color: "#4b5563" }}>
          <span
            className="h-1.5 w-1.5 rounded-full animate-pulse"
            style={{ backgroundColor: meta.color }}
          />
          Analyzing specimen…
        </div>
      )}
    </div>
  );
}
