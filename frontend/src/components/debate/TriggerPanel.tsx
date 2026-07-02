"use client";

/**
 * TriggerPanel — compact inline metric row for the divergence gate.
 *
 * Replaces the previous elaborate radial SVG gauge + entropy bars with a
 * minimal horizontal strip showing JS divergence, per-agent entropy, and
 * the fired/not-fired status. Clinical, data-dense, zero ornamentation.
 */

import type { TriggerResult } from "@/types/debate";
import { useCountup } from "@/hooks/useCountup";

interface TriggerPanelProps {
  trigger: TriggerResult | null;
}

function MetricCell({
  label,
  value,
  unit,
  color,
}: {
  label: string;
  value: string;
  unit?: string;
  color?: string;
}): React.JSX.Element {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="font-mono text-[9px] uppercase tracking-widest" style={{ color: "#6b7280" }}>
        {label}
      </span>
      <span className="font-mono text-sm tabular" style={{ color: color ?? "#e5e7eb" }}>
        {value}
        {unit && <span style={{ color: "#4b5563" }}>{unit}</span>}
      </span>
    </div>
  );
}

export default function TriggerPanel({ trigger }: TriggerPanelProps): React.JSX.Element {
  if (!trigger) {
    return (
      <div className="flex items-center gap-2 px-5 py-3 font-mono text-[11px]" style={{ color: "#4b5563" }}>
        <span className="h-1.5 w-1.5 rounded-full animate-pulse" style={{ backgroundColor: "#3b82f6" }} />
        Evaluating divergence…
      </div>
    );
  }

  const js = useCountup(trigger.js_divergence, { duration: 800 });
  const entA = useCountup(trigger.entropy_a, { duration: 800 });
  const entB = useCountup(trigger.entropy_b, { duration: 800 });

  const fired = trigger.fired;

  return (
    <div
      className="flex flex-wrap items-center gap-6 px-5 py-3"
      style={{
        borderTop: "1px solid #1a1a1f",
        borderBottom: "1px solid #1a1a1f",
      }}
    >
      {/* Status indicator */}
      <div className="flex items-center gap-2">
        <span
          className="h-2 w-2 rounded-full"
          style={{
            backgroundColor: fired ? "#fbbf24" : "#059669",
          }}
        />
        <span
          className="font-mono text-[11px] font-medium uppercase tracking-widest"
          style={{ color: fired ? "#fbbf24" : "#059669" }}
        >
          {fired ? "Debate triggered" : "Fast path"}
        </span>
      </div>

      {/* Separator */}
      <div className="hidden h-6 w-px sm:block" style={{ backgroundColor: "#1f1f23" }} />

      {/* Metrics */}
      <div className="flex flex-wrap items-center gap-5">
        <MetricCell
          label="JS Divergence"
          value={js.toFixed(4)}
          color={fired ? "#fbbf24" : "#e5e7eb"}
        />
        <MetricCell
          label={`Threshold`}
          value={trigger.threshold_js.toFixed(2)}
        />
        <MetricCell
          label="Entropy A"
          value={entA.toFixed(3)}
          unit=" bits"
        />
        <MetricCell
          label="Entropy B"
          value={entB.toFixed(3)}
          unit=" bits"
        />
        <MetricCell
          label="Entropy Threshold"
          value={trigger.threshold_entropy.toFixed(2)}
        />
      </div>
    </div>
  );
}
