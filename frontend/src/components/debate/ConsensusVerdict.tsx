"use client";

/**
 * ConsensusVerdict — the resolution. Rises into view when the calibrated fusion
 * lands: the diagnosis types in, the confidence counts up behind a growing bar,
 * a calibration badge grades the ECE, the full distribution settles, and the
 * neutral Consensus-Engine synthesis streams in beneath it in italic serif.
 */

import type { ConsensusResult, TriggerResult } from "@/types/debate";
import { getClassMeta, getRisk } from "@/lib/constants";
import { useCountup } from "@/hooks/useCountup";
import ArgumentStream from "@/components/debate/ArgumentStream";
import ProbabilityBars from "@/components/debate/ProbabilityBars";

interface ConsensusVerdictProps {
  consensus: ConsensusResult;
  trigger: TriggerResult | null;
  synthesis: string;
  synthesisActive: boolean;
}

interface Calibration {
  label: string;
  color: string;
}

function calibration(ece: number): Calibration {
  if (ece < 0.05) return { label: "Well calibrated", color: "var(--accent-consensus)" };
  if (ece <= 0.1) return { label: "Moderately calibrated", color: "var(--accent-warning)" };
  return { label: "Poorly calibrated", color: "var(--accent-danger)" };
}

export default function ConsensusVerdict({
  consensus,
  trigger,
  synthesis,
  synthesisActive,
}: ConsensusVerdictProps): React.JSX.Element {
  const meta = getClassMeta(consensus.pred_class);
  const fullName = meta?.fullName ?? consensus.pred_class;
  const riskColor = getRisk(consensus.pred_class);
  const conf = useCountup(consensus.confidence * 100, { duration: 1200, delay: 200 });
  const cal = calibration(consensus.ece);
  const pathLabel = trigger?.fired
    ? "Resolved through adversarial spatial debate"
    : "Fast path — the agents agreed";

  return (
    <section
      aria-label="Consensus diagnosis"
      className="animate-rise-in overflow-hidden rounded-2xl border bg-surface p-7 shadow-glow-consensus"
      style={{ borderColor: "rgba(5, 150, 105, 0.4)" }}
    >
      <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-consensus">
        <span aria-hidden>✦</span>
        Consensus diagnosis
      </div>

      {/* Diagnosis + confidence */}
      <div className="mt-4 flex flex-wrap items-end justify-between gap-6">
        <div>
          <h2 className="font-display text-3xl leading-none text-ink sm:text-[38px]">
            {fullName}
          </h2>
          <div className="mt-2 flex items-center gap-2">
            <span
              className="rounded-md px-2 py-0.5 font-mono text-xs font-semibold text-white"
              style={{ backgroundColor: riskColor }}
            >
              {consensus.pred_class}
            </span>
            <span className="text-xs text-ink-soft">{pathLabel}</span>
          </div>
        </div>
        <div className="text-right">
          <div className="text-[10px] uppercase tracking-wider text-ink-faint">
            Confidence
          </div>
          <div className="font-mono text-3xl font-semibold tabular leading-none text-consensus">
            {conf.toFixed(0)}
            <span className="text-xl text-ink-faint">%</span>
          </div>
        </div>
      </div>

      {/* Confidence bar */}
      <div className="mt-4 h-2.5 w-full overflow-hidden rounded-full bg-surface-alt">
        <div
          className="h-full rounded-full"
          style={{
            width: `${conf}%`,
            background:
              "linear-gradient(90deg, var(--accent-consensus) 0%, #34d399 100%)",
          }}
        />
      </div>

      {/* Metrics */}
      <div className="mt-5 flex flex-wrap gap-2.5">
        <Metric label="Temperature" value={consensus.temperature.toFixed(2)} />
        <Metric label="ECE" value={consensus.ece.toFixed(3)} />
        <span
          className="inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium"
          style={{ color: cal.color, borderColor: cal.color }}
        >
          <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: cal.color }} />
          {cal.label}
        </span>
      </div>

      {/* Final distribution */}
      <div className="mt-6">
        <div className="mb-2 text-[10px] uppercase tracking-wider text-ink-faint">
          Final calibrated distribution
        </div>
        <ProbabilityBars
          probabilities={consensus.probabilities}
          color="#059669"
          predClass={consensus.pred_class}
        />
      </div>

      {/* Synthesis */}
      {(synthesis.length > 0 || synthesisActive) && (
        <div className="mt-6 rounded-xl border border-hairline bg-surface-alt p-4">
          <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-consensus">
            Argus Consensus Engine
          </div>
          <p className="text-[15px] leading-relaxed text-ink-soft">
            <ArgumentStream text={synthesis} active={synthesisActive} serif speed={42} />
          </p>
        </div>
      )}

      <p className="mt-6 font-mono text-[11px] text-ink-faint">
        Research prototype — not for clinical use.
      </p>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }): React.JSX.Element {
  return (
    <span className="inline-flex items-center gap-2 rounded-lg border border-hairline bg-surface-alt px-3 py-1.5">
      <span className="text-[10px] uppercase tracking-wider text-ink-faint">{label}</span>
      <span className="font-mono text-xs font-medium tabular text-ink">{value}</span>
    </span>
  );
}
