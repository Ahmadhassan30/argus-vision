"use client";

/**
 * TriggerPanel — the divergence/entropy gate that decides whether the cheap
 * "fast path" suffices or the full adversarial debate must fire. On the left a
 * radial gauge plots the Jensen–Shannon divergence between the two agents'
 * distributions against its firing threshold; on the right, paired horizontal
 * bars show each agent's Shannon entropy (uncertainty) against the entropy
 * threshold. Numbers count up with a back-out "snap" while the panel itself
 * glows amber when the debate is triggered. A fixed min-height loading state
 * keeps the layout perfectly still until the verdict lands.
 */

import type { TriggerResult } from "@/types/debate";
import { useCountup, easeOutBack } from "@/hooks/useCountup";
import { AGENT_A, AGENT_B, COLORS } from "@/lib/constants";

/** Maximum Shannon entropy over 8 classes: log2(8) = 3 bits. */
const ENTROPY_MAX = 3;

/** Gauge geometry (SVG user units). */
const GAUGE = { size: 140, cx: 70, cy: 70, r: 52, stroke: 12 } as const;
const CIRC = 2 * Math.PI * GAUGE.r;

/** Point on a circle for a 0..1 fraction, starting at 12 o'clock, clockwise. */
function polar(r: number, fraction: number): { x: number; y: number } {
  const rad = (fraction * 360 - 90) * (Math.PI / 180);
  return { x: GAUGE.cx + r * Math.cos(rad), y: GAUGE.cy + r * Math.sin(rad) };
}

interface TriggerPanelProps {
  trigger: TriggerResult | null;
}

/** One agent's animated entropy bar with an inline threshold marker. */
function EntropyBar({
  label,
  value,
  color,
  threshold,
}: {
  label: string;
  value: number;
  color: string;
  threshold: number;
}): React.JSX.Element {
  const animated = useCountup(value, { duration: 1000 });
  const widthPct = Math.max(0, Math.min(100, (animated / ENTROPY_MAX) * 100));
  const threshPct = Math.max(0, Math.min(100, (threshold / ENTROPY_MAX) * 100));

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <span
          className="text-[11px] font-medium uppercase tracking-[0.12em]"
          style={{ color }}
        >
          {label}
        </span>
        <span className="font-mono text-sm tabular text-ink">
          {animated.toFixed(2)}
          <span className="text-ink-faint"> bits</span>
        </span>
      </div>
      <div
        className="relative h-3 w-full overflow-hidden rounded-full bg-surface-alt"
        role="progressbar"
        aria-label={`${label} Shannon entropy`}
        aria-valuemin={0}
        aria-valuemax={ENTROPY_MAX}
        aria-valuenow={Number(value.toFixed(2))}
      >
        <div
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ width: `${widthPct}%`, backgroundColor: color }}
        />
        {/* Threshold line, drawn over the fill so it stays visible. */}
        <div
          className="absolute inset-y-0 w-px"
          style={{ left: `${threshPct}%`, backgroundColor: COLORS.ink }}
        />
      </div>
    </div>
  );
}

export default function TriggerPanel({ trigger }: TriggerPanelProps): React.JSX.Element {
  // Hooks run unconditionally (stable order); targets are 0 until the result lands.
  const jsValue = useCountup(trigger?.js_divergence ?? 0, {
    duration: 1000,
    easing: easeOutBack,
  });

  // ----- Loading: trigger not yet evaluated. -----
  if (trigger === null) {
    return (
      <div
        role="region"
        aria-label="Debate trigger — evaluating"
        aria-busy="true"
        className="rounded-2xl border border-hairline bg-surface p-6 shadow-panel animate-panel-enter"
      >
        <header className="flex items-center justify-between">
          <div>
            <div className="text-[11px] font-medium uppercase tracking-[0.12em] text-ink-faint">
              Divergence gate
            </div>
            <h3 className="font-display text-xl leading-tight text-ink">Debate trigger</h3>
          </div>
        </header>

        <div className="mt-5 flex min-h-[220px] flex-col gap-6">
          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            {/* Gauge skeleton. */}
            <div className="flex items-center justify-center">
              <div className="shimmer animate-shimmer h-[140px] w-[140px] rounded-full" />
            </div>
            {/* Bars skeleton. */}
            <div className="flex flex-col justify-center gap-5">
              <div className="space-y-2">
                <div className="shimmer animate-shimmer h-3 w-24 rounded" />
                <div className="shimmer animate-shimmer h-3 w-full rounded-full" />
              </div>
              <div className="space-y-2">
                <div className="shimmer animate-shimmer h-3 w-24 rounded" />
                <div className="shimmer animate-shimmer h-3 w-full rounded-full" />
              </div>
            </div>
          </div>

          <div className="mt-auto flex items-center gap-2.5 text-xs text-ink-faint">
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-hairline border-t-warning" />
            Evaluating divergence…
          </div>
        </div>
      </div>
    );
  }

  // ----- Evaluated. -----
  const fired = trigger.fired;
  const jsHot = trigger.js_divergence >= trigger.threshold_js;
  const arcColor = jsHot ? COLORS.warning : COLORS.consensus;
  const arcFrac = Math.max(0, Math.min(1, jsValue));

  const tick = {
    inner: polar(GAUGE.r - GAUGE.stroke / 2 - 3, trigger.threshold_js),
    outer: polar(GAUGE.r + GAUGE.stroke / 2 + 3, trigger.threshold_js),
  };

  return (
    <div
      role="region"
      aria-label="Debate trigger evaluation"
      className={[
        "rounded-2xl border border-hairline bg-surface p-6 animate-panel-enter",
        fired ? "shadow-glow-warning" : "shadow-panel",
      ].join(" ")}
      style={{ borderColor: fired ? COLORS.warning : undefined }}
    >
      {/* Header. */}
      <header className="flex items-start justify-between gap-4">
        <div>
          <div className="text-[11px] font-medium uppercase tracking-[0.12em] text-ink-faint">
            Divergence gate
          </div>
          <h3 className="font-display text-xl leading-tight text-ink">Debate trigger</h3>
        </div>
        {fired ? (
          <span
            className="flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.12em]"
            style={{ backgroundColor: "rgba(217, 119, 6, 0.12)", color: COLORS.warning }}
          >
            <span
              className="h-2 w-2 rounded-full animate-pulse-dot-fast"
              style={{ backgroundColor: COLORS.warning }}
            />
            Fired
          </span>
        ) : (
          <span
            className="flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.12em]"
            style={{ backgroundColor: "rgba(5, 150, 105, 0.12)", color: COLORS.consensus }}
          >
            <span
              className="h-2 w-2 rounded-full"
              style={{ backgroundColor: COLORS.consensus }}
            />
            Fast path
          </span>
        )}
      </header>

      {/* Two-column body. */}
      <div className="mt-5 grid min-h-[220px] grid-cols-1 gap-8 md:grid-cols-2">
        {/* LEFT — Jensen–Shannon divergence radial gauge. */}
        <section className="flex flex-col items-center">
          <div className="mb-3 w-full text-[11px] font-medium uppercase tracking-[0.12em] text-ink-soft">
            Jensen–Shannon divergence
          </div>
          <div className="relative" style={{ width: GAUGE.size, height: GAUGE.size }}>
            <svg
              width={GAUGE.size}
              height={GAUGE.size}
              viewBox={`0 0 ${GAUGE.size} ${GAUGE.size}`}
              role="progressbar"
              aria-label="Jensen–Shannon divergence"
              aria-valuemin={0}
              aria-valuemax={1}
              aria-valuenow={Number(trigger.js_divergence.toFixed(3))}
            >
              {/* Track ring. */}
              <circle
                cx={GAUGE.cx}
                cy={GAUGE.cy}
                r={GAUGE.r}
                fill="none"
                stroke={COLORS.surfaceAlt}
                strokeWidth={GAUGE.stroke}
              />
              {/* Value arc — starts at 12 o'clock, sweeps clockwise. */}
              <circle
                cx={GAUGE.cx}
                cy={GAUGE.cy}
                r={GAUGE.r}
                fill="none"
                stroke={arcColor}
                strokeWidth={GAUGE.stroke}
                strokeLinecap="round"
                strokeDasharray={CIRC}
                strokeDashoffset={CIRC * (1 - arcFrac)}
                transform={`rotate(-90 ${GAUGE.cx} ${GAUGE.cy})`}
              />
              {/* Threshold tick. */}
              <line
                x1={tick.inner.x}
                y1={tick.inner.y}
                x2={tick.outer.x}
                y2={tick.outer.y}
                stroke={COLORS.ink}
                strokeWidth={2}
                strokeLinecap="round"
              />
            </svg>
            {/* Centered numeric readout. */}
            <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
              <span
                className="font-mono text-2xl font-medium tabular leading-none"
                style={{ color: arcColor }}
              >
                {jsValue.toFixed(3)}
              </span>
            </div>
          </div>
          <div className="mt-3 font-mono text-[11px] tabular text-ink-faint">
            threshold {trigger.threshold_js.toFixed(2)}
          </div>
        </section>

        {/* RIGHT — Shannon entropy bars. */}
        <section className="flex flex-col justify-center">
          <div className="mb-3 text-[11px] font-medium uppercase tracking-[0.12em] text-ink-soft">
            Shannon entropy
          </div>
          <div className="flex flex-col gap-5">
            <EntropyBar
              label={AGENT_A.label}
              value={trigger.entropy_a}
              color={AGENT_A.color}
              threshold={trigger.threshold_entropy}
            />
            <EntropyBar
              label={AGENT_B.label}
              value={trigger.entropy_b}
              color={AGENT_B.color}
              threshold={trigger.threshold_entropy}
            />
          </div>
          <div className="mt-4 flex items-center gap-2 font-mono text-[11px] tabular text-ink-faint">
            <span className="inline-block h-3 w-px bg-ink" aria-hidden />
            threshold {trigger.threshold_entropy.toFixed(2)} bits
          </div>
        </section>
      </div>
    </div>
  );
}
