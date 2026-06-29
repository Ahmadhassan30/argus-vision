"use client";

/**
 * TimelineRail — the vertical "procedure" pipeline. It renders the nine
 * milestones of a classification job as a single connected rail: a circular
 * node per step, a connector line that fills with colour as progress flows
 * downward, the step label (with an optional agent eyebrow), and a relative
 * "+X.Xs" timestamp once the step lands.
 *
 * Two steps are dynamic. Step 6 ("path") reflects the routing decision —
 * "Debate triggered" (amber), "Fast consensus" (emerald) or "Routing…"
 * (muted) — and step 7 ("attention") collapses to a struck-through, hollow
 * "skipped" node whenever the fast path was taken. The first not-yet-completed
 * *reachable* step is marked active with a gentle accent pulse. All motion is
 * pure CSS (transition / animate-pulse-dot) so the global prefers-reduced-motion
 * rule neutralises it for free.
 */

import { COLORS } from "@/lib/constants";
import type { StepKey, UiPhase } from "@/lib/debateReducer";

interface TimelineRailProps {
  /** Epoch-ms completion time for each reached milestone. */
  completedAt: Partial<Record<StepKey, number>>;
  /** Coarse UI phase (used to suppress the active pulse on error). */
  phase: UiPhase;
  /** Whether the divergence trigger fired; null until routing resolves. */
  triggerFired: boolean | null;
}

/** Visual state of a single node. */
type NodeStatus = "completed" | "active" | "pending" | "skipped";

/** Static blueprint for the nine pipeline milestones (in order). */
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

/** The circular marker at the head of each row. */
function StepNode({ status, accent }: { status: NodeStatus; accent: string }): React.JSX.Element {
  if (status === "completed") {
    return (
      <span
        className="relative flex h-[18px] w-[18px] shrink-0 items-center justify-center rounded-full transition-all duration-300"
        style={{ backgroundColor: accent, boxShadow: `0 0 0 3px ${accent}1f` }}
      >
        <svg viewBox="0 0 24 24" className="h-2.5 w-2.5" fill="none" aria-hidden>
          <path
            d="M5 13l4 4L19 7"
            stroke="#fff"
            strokeWidth={3}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </span>
    );
  }

  if (status === "active") {
    return (
      <span
        className="relative flex h-[18px] w-[18px] shrink-0 items-center justify-center rounded-full border-[1.5px] bg-surface transition-all duration-300"
        style={{ borderColor: accent }}
      >
        <span
          className="absolute inset-0 rounded-full animate-pulse-dot"
          style={{ backgroundColor: `${accent}33` }}
          aria-hidden
        />
        <span
          className="relative h-1.5 w-1.5 rounded-full"
          style={{ backgroundColor: accent }}
          aria-hidden
        />
      </span>
    );
  }

  if (status === "skipped") {
    return (
      <span
        className="flex h-[18px] w-[18px] shrink-0 rounded-full border-[1.5px] border-dashed border-ink-faint bg-surface transition-all duration-300"
        aria-hidden
      />
    );
  }

  // pending
  return (
    <span
      className="h-[18px] w-[18px] shrink-0 rounded-full border-[1.5px] border-hairline bg-surface transition-all duration-300"
      aria-hidden
    />
  );
}

export default function TimelineRail({
  completedAt,
  phase,
  triggerFired,
}: TimelineRailProps): React.JSX.Element {
  // Resolve dynamic labels / accents and per-step status flags.
  const steps = BASE_STEPS.map((s) => {
    let label = s.label;
    let accent = s.accent ?? COLORS.ink;
    let skipped = false;

    if (s.key === "path") {
      if (triggerFired === true) {
        label = "Debate triggered";
        accent = COLORS.warning;
      } else if (triggerFired === false) {
        label = "Fast consensus";
        accent = COLORS.consensus;
      } else {
        label = "Routing…";
        accent = COLORS.inkFaint;
      }
    }

    if (s.key === "attention" && triggerFired === false) {
      skipped = true;
      label = "Spatial attention — skipped";
      accent = COLORS.inkFaint;
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

  // Timeline origin: upload time, else the earliest stamp we have.
  const stamps = Object.values(completedAt).filter((v): v is number => typeof v === "number");
  const base =
    completedAt.uploaded ?? (stamps.length > 0 ? Math.min(...stamps) : undefined);

  // The first not-yet-completed, non-skipped step pulses as "active".
  const activeIndex =
    phase === "error" ? -1 : steps.findIndex((s) => !s.completed && !s.skipped);

  return (
    <section
      aria-label="Pipeline timeline"
      className="w-full rounded-2xl border border-hairline bg-surface p-6 shadow-panel animate-panel-enter"
    >
      <header className="mb-5">
        <div className="text-[11px] font-medium uppercase tracking-[0.12em] text-ink-faint">
          Pipeline
        </div>
        <h2 className="mt-0.5 font-display text-xl leading-tight text-ink">Procedure</h2>
      </header>

      <ol className="relative">
        {steps.map((s, i) => {
          const isLast = i === steps.length - 1;
          const status: NodeStatus = s.completed
            ? "completed"
            : s.skipped
              ? "skipped"
              : i === activeIndex
                ? "active"
                : "pending";

          // The connector below this node fills once the NEXT step lands.
          const next = steps[i + 1];
          const connectorColor =
            next && next.completed ? next.accent : COLORS.hairline;

          const labelClass =
            status === "completed"
              ? "text-sm text-ink"
              : status === "active"
                ? "text-sm font-medium text-ink"
                : status === "skipped"
                  ? "text-sm text-ink-faint line-through"
                  : "text-sm text-ink-faint";

          return (
            <li key={s.key} className="flex gap-3.5">
              {/* Node + connector track. */}
              <div className="flex flex-col items-center">
                <StepNode status={status} accent={s.accent} />
                {!isLast && (
                  <span
                    className="w-[2px] flex-1 rounded-full transition-all duration-300"
                    style={{ backgroundColor: connectorColor, minHeight: 22 }}
                  />
                )}
              </div>

              {/* Label, eyebrow, timestamp. */}
              <div
                className={[
                  "flex flex-1 items-start justify-between gap-3",
                  isLast ? "pb-0" : "pb-6",
                ].join(" ")}
              >
                <div className="min-w-0">
                  {s.eyebrow && (
                    <div
                      className="text-[10px] font-medium uppercase tracking-[0.12em]"
                      style={{ color: s.accent }}
                    >
                      {s.eyebrow}
                    </div>
                  )}
                  <div className={labelClass}>{s.label}</div>
                </div>

                {s.completed && base !== undefined && (
                  <div className="shrink-0 pt-px font-mono text-[11px] tabular text-ink-faint">
                    +{((completedAt[s.key]! - base) / 1000).toFixed(1)}s
                  </div>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
