"use client";

/**
 * ProbabilityBars — the eight ISIC classes as animated horizontal bars in
 * canonical order. Each fill grows from 0 to its value on entry and transitions
 * smoothly if the distribution updates; the leading class row is subtly
 * highlighted. Numbers are tabular monospace so columns stay aligned.
 */

import { useEffect, useState } from "react";

import { CLASS_ORDER, getClassName } from "@/lib/constants";

interface ProbabilityBarsProps {
  probabilities: Record<string, number>;
  /** Bar colour (agent accent or consensus emerald). */
  color: string;
  /** Class id to highlight; defaults to the argmax of `probabilities`. */
  predClass?: string;
  /** Show only the top-N classes by probability (default: all 8). */
  max?: number;
  className?: string;
}

function argmax(probs: Record<string, number>): string {
  let best = CLASS_ORDER[0];
  let bestV = -Infinity;
  for (const cls of CLASS_ORDER) {
    const v = probs[cls] ?? 0;
    if (v > bestV) {
      bestV = v;
      best = cls;
    }
  }
  return best;
}

export default function ProbabilityBars({
  probabilities,
  color,
  predClass,
  max,
  className = "",
}: ProbabilityBarsProps): React.JSX.Element {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setMounted(true));
    return () => cancelAnimationFrame(id);
  }, []);

  const leader = predClass ?? argmax(probabilities);

  // Full canonical order, or just the top-N (still rendered highest-first).
  const classes =
    max && max < CLASS_ORDER.length
      ? [...CLASS_ORDER].sort((a, b) => (probabilities[b] ?? 0) - (probabilities[a] ?? 0)).slice(0, max)
      : CLASS_ORDER;

  return (
    <div className={["flex flex-col gap-1.5", className].join(" ")}>
      {classes.map((cls) => {
        const p = probabilities[cls] ?? 0;
        const pct = Math.round(p * 1000) / 10;
        const isLeader = cls === leader;
        return (
          <div
            key={cls}
            className={[
              "group grid grid-cols-[44px_1fr_46px] items-center gap-2.5 rounded-md px-1.5 py-1 transition-colors",
              isLeader ? "bg-surface-alt" : "",
            ].join(" ")}
          >
            <span
              className="font-mono text-[11px] font-medium tracking-wide"
              style={{ color: isLeader ? color : "var(--text-secondary)" }}
              title={getClassName(cls)}
            >
              {cls}
            </span>
            <div
              className="relative h-2 overflow-hidden rounded-full bg-surface-alt"
              role="progressbar"
              aria-valuenow={pct}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label={`${getClassName(cls)} probability`}
            >
              <div
                className="absolute inset-y-0 left-0 rounded-full"
                style={{
                  width: mounted ? `${p * 100}%` : "0%",
                  background: `linear-gradient(90deg, ${color} 0%, ${color}cc 65%, ${color}40 100%)`,
                  transition:
                    "width 650ms cubic-bezier(0.16, 1, 0.3, 1)",
                  opacity: isLeader ? 1 : 0.78,
                }}
              />
            </div>
            <span
              className="text-right font-mono text-[11px] tabular text-ink-soft"
              style={{ color: isLeader ? "var(--text-primary)" : undefined }}
            >
              {pct.toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}
