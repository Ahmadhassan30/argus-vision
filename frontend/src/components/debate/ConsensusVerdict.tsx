"use client";

/**
 * ConsensusVerdict — the final diagnosis panel.
 *
 * Flat, clean panel with a subtle green left accent stripe. No glow, no
 * rise-in animation, no sparkle icons. Shows the diagnosis, confidence bar,
 * calibration metrics, full probability distribution, and synthesis text.
 */

import type { ConsensusResult, TriggerResult } from "@/types/debate";
import { getClassMeta, getRisk } from "@/lib/constants";
import { useCountup } from "@/hooks/useCountup";
import ProbabilityBars from "@/components/debate/ProbabilityBars";

interface ConsensusVerdictProps {
  consensus: ConsensusResult;
  trigger: TriggerResult | null;
  synthesis: string;
  synthesisActive: boolean;
}

function calibrationLabel(ece: number): { label: string; color: string } {
  if (ece < 0.05) return { label: "Well calibrated", color: "#059669" };
  if (ece <= 0.1) return { label: "Moderately calibrated", color: "#fbbf24" };
  return { label: "Poorly calibrated", color: "#f87171" };
}

export default function ConsensusVerdict({
  consensus,
  trigger,
  synthesis,
}: ConsensusVerdictProps): React.JSX.Element {
  const meta = getClassMeta(consensus.pred_class);
  const fullName = meta?.fullName ?? consensus.pred_class;
  const riskColor = getRisk(consensus.pred_class);
  const conf = useCountup(consensus.confidence * 100, { duration: 1200, delay: 200 });
  const cal = calibrationLabel(consensus.ece);
  const pathLabel = trigger?.fired
    ? "Resolved through adversarial spatial debate"
    : "Fast path — agents agreed";

  return (
    <div className="p-5" style={{ borderLeft: "3px solid #059669" }}>
      {/* Section label */}
      <div className="font-mono text-[10px] font-medium uppercase tracking-widest" style={{ color: "#059669" }}>
        Consensus Diagnosis
      </div>

      {/* Diagnosis + confidence */}
      <div className="mt-3 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-2xl font-semibold leading-none" style={{ color: "#f3f4f6" }}>
            {fullName}
          </h2>
          <div className="mt-2 flex items-center gap-2">
            <span
              className="rounded px-2 py-0.5 font-mono text-[11px] font-semibold text-white"
              style={{ backgroundColor: riskColor }}
            >
              {consensus.pred_class}
            </span>
            <span className="font-mono text-[10px]" style={{ color: "#6b7280" }}>
              {pathLabel}
            </span>
          </div>
        </div>
        <div className="text-right">
          <div className="font-mono text-[10px] uppercase tracking-wider" style={{ color: "#6b7280" }}>
            Confidence
          </div>
          <div className="font-mono text-2xl font-semibold tabular leading-none" style={{ color: "#059669" }}>
            {conf.toFixed(0)}
            <span className="text-base" style={{ color: "#6b7280" }}>%</span>
          </div>
        </div>
      </div>

      {/* Confidence bar */}
      <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full" style={{ backgroundColor: "#1a1a1f" }}>
        <div
          className="h-full rounded-full transition-[width] duration-1000"
          style={{
            width: `${conf}%`,
            backgroundColor: "#059669",
          }}
        />
      </div>

      {/* Metrics row */}
      <div className="mt-4 flex flex-wrap items-center gap-4">
        <MetricChip label="Temperature" value={consensus.temperature.toFixed(2)} />
        <MetricChip label="ECE" value={consensus.ece.toFixed(3)} />
        <div className="flex items-center gap-1.5 font-mono text-[11px]" style={{ color: cal.color }}>
          <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: cal.color }} />
          {cal.label}
        </div>
      </div>

      {/* Final distribution */}
      <div className="mt-5">
        <div className="mb-2 font-mono text-[10px] uppercase tracking-widest" style={{ color: "#6b7280" }}>
          Final calibrated distribution
        </div>
        <ProbabilityBars
          probabilities={consensus.probabilities}
          color="#059669"
          predClass={consensus.pred_class}
        />
      </div>

      {/* Synthesis */}
      {synthesis.length > 0 && (
        <div
          className="mt-5 rounded border p-4"
          style={{ borderColor: "#1a1a1f", backgroundColor: "#0a0a0c" }}
        >
          <div className="mb-1 font-mono text-[9px] font-medium uppercase tracking-widest" style={{ color: "#059669" }}>
            Consensus Engine
          </div>
          <p className="text-[13px] leading-relaxed" style={{ color: "#9ca3af" }}>
            {synthesis}
          </p>
        </div>
      )}

      <p className="mt-4 font-mono text-[10px]" style={{ color: "#4b5563" }}>
        Research prototype — not for clinical use.
      </p>
    </div>
  );
}

function MetricChip({ label, value }: { label: string; value: string }): React.JSX.Element {
  return (
    <span
      className="inline-flex items-center gap-2 rounded border px-2.5 py-1"
      style={{ borderColor: "#1a1a1f", backgroundColor: "#0a0a0c" }}
    >
      <span className="font-mono text-[9px] uppercase tracking-wider" style={{ color: "#6b7280" }}>
        {label}
      </span>
      <span className="font-mono text-xs tabular" style={{ color: "#e5e7eb" }}>
        {value}
      </span>
    </span>
  );
}
