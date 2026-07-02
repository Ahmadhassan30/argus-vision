"use client";

/**
 * TimelineRail — compact vertical pipeline timeline.
 *
 * Smaller 12px nodes, thinner connector lines, no background panel or shadow.
 * Sits in the sidebar and communicates pipeline progress at a glance with
 * minimal visual weight.
 */

import { COLORS } from "@/lib/constants";
import type { StepKey, UiPhase } from "@/lib/debateReducer";

interface TimelineRailProps {
  completedAt: Partial<Record<StepKey, number>>;
  phase: UiPhase;
  triggerFired: boolean | null;
}

type NodeStatus = "completed" | "active" | "pending" | "skipped";

const BASE_STEPS: ReadonlyArray<{
  key: StepKey;
  label: string;
  eyebrow?: string;
  accent?: string;
}> = [
  { key: "uploaded", label: "Image received" },
  { key: "agents_init", label: "Agents initialized" },
  { key: "analysis_a", label: "CNN analysis", eyebrow: "Agent A", accent: COLORS.agentA },
  { key: "analysis_b", label: "ViT analysis", eyebrow: "Agent B", accent: COLORS.agentB },
  { key: "divergence", label: "Divergence computed" },
  { key: "path", label: "Routing…" },
  { key: "attention", label: "Spatial attention" },
  { key: "consensus", label: "Consensus fusion", accent: COLORS.consensus },
  { key: "delivered", label: "Diagnosis delivered", accent: COLORS.consensus },
];

function StepNode({ status, accent }: { status: NodeStatus; accent: string }): React.JSX.Element {
  const size = "h-3 w-3"; // 12px

  if (status === "completed") {
    return (
      <span
        className={`relative flex ${size} shrink-0 items-center justify-center rounded-full`}
        style={{ backgroundColor: accent }}
      >
        <svg viewBox="0 0 24 24" className="h-2 w-2" fill="none" aria-hidden>
          <path d="M5 13l4 4L19 7" stroke="#fff" strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </span>
    );
  }

  if (status === "active") {
    return (
      <span
        className={`relative flex ${size} shrink-0 items-center justify-center rounded-full border`}
        style={{ borderColor: accent }}
      >
        <span
          className="h-1 w-1 rounded-full animate-pulse"
          style={{ backgroundColor: accent }}
        />
      </span>
    );
  }

  if (status === "skipped") {
    return (
      <span
        className={`${size} shrink-0 rounded-full border border-dashed`}
        style={{ borderColor: "#4b5563" }}
      />
    );
  }

  // pending
  return (
    <span
      className={`${size} shrink-0 rounded-full border`}
      style={{ borderColor: "#1f1f23" }}
    />
  );
}

export default function TimelineRail({
  completedAt,
  phase,
  triggerFired,
}: TimelineRailProps): React.JSX.Element {
  const steps = BASE_STEPS.map((s) => {
    let label = s.label;
    let accent = s.accent ?? "#9ca3af";
    let skipped = false;

    if (s.key === "path") {
      if (triggerFired === true) {
        label = "Debate triggered";
        accent = "#fbbf24";
      } else if (triggerFired === false) {
        label = "Fast consensus";
        accent = "#059669";
      } else {
        label = "Routing…";
        accent = "#6b7280";
      }
    }

    if (s.key === "attention" && triggerFired === false) {
      skipped = true;
      label = "Attention — skipped";
      accent = "#6b7280";
    }

    return {
      key: s.key,
      eyebrow: s.eyebrow,
      label,
      accent,
      skipped,
      completed: completedAt[s.key] !== undefined,
    };
  });

  const stamps = Object.values(completedAt).filter((v): v is number => typeof v === "number");
  const base = completedAt.uploaded ?? (stamps.length > 0 ? Math.min(...stamps) : undefined);

  const activeIndex =
    phase === "error" ? -1 : steps.findIndex((s) => !s.completed && !s.skipped);

  return (
    <div>
      <div className="mb-3 font-mono text-[10px] uppercase tracking-widest" style={{ color: "#6b7280" }}>
        Pipeline
      </div>

      <ol className="relative flex flex-col">
        {steps.map((s, i) => {
          const isLast = i === steps.length - 1;
          const status: NodeStatus = s.completed
            ? "completed"
            : s.skipped
              ? "skipped"
              : i === activeIndex
                ? "active"
                : "pending";

          const next = steps[i + 1];
          const connectorColor = next && next.completed ? next.accent : "#1f1f23";

          return (
            <li key={s.key} className="flex gap-3">
              {/* Node + connector */}
              <div className="flex flex-col items-center">
                <StepNode status={status} accent={s.accent} />
                {!isLast && (
                  <span
                    className="w-px flex-1"
                    style={{ backgroundColor: connectorColor, minHeight: 18 }}
                  />
                )}
              </div>

              {/* Label + timestamp */}
              <div className={["flex flex-1 items-start justify-between gap-2", isLast ? "pb-0" : "pb-4"].join(" ")}>
                <div className="min-w-0">
                  {s.eyebrow && (
                    <div
                      className="font-mono text-[9px] uppercase tracking-widest"
                      style={{ color: s.accent }}
                    >
                      {s.eyebrow}
                    </div>
                  )}
                  <div
                    className="text-[12px]"
                    style={{
                      color: status === "completed" || status === "active" ? "#e5e7eb" : "#4b5563",
                      textDecoration: status === "skipped" ? "line-through" : "none",
                    }}
                  >
                    {s.label}
                  </div>
                </div>

                {s.completed && base !== undefined && (
                  <span className="shrink-0 font-mono text-[10px] tabular" style={{ color: "#4b5563" }}>
                    +{((completedAt[s.key]! - base) / 1000).toFixed(1)}s
                  </span>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
