"use client";

/**
 * DisagreementMap visualizes the contested region between the two agents. It
 * renders the disagreement heatmap (with the contested bounding box overlaid
 * via {@link HeatmapCanvas}) and, below it, a comparison of the mean activation
 * inside the contested region for Agent A versus Agent B. When no attention
 * result is available it renders a faint placeholder.
 */

import { motion } from "framer-motion";
import type { AttentionResult } from "@/types/debate";
import { AGENT_A_COLOR, AGENT_B_COLOR } from "@/lib/constants";
import HeatmapCanvas from "@/components/debate/HeatmapCanvas";

/** The region-stats key holding the mean activation in the contested region. */
const MEAN_KEY = "mean";

/** Props for {@link DisagreementMap}. */
export interface DisagreementMapProps {
  /** The spatial attention result, or null before it is computed. */
  attention: AttentionResult | null;
}

/**
 * Extracts a representative scalar (the contested-region mean activation) from
 * a region-stats map. Falls back to the first numeric value, then to 0.
 *
 * @param stats - The per-agent region statistics map.
 * @returns The mean activation in [0, ...], defaulting to 0 when absent.
 */
function regionMean(stats: Record<string, number>): number {
  if (typeof stats[MEAN_KEY] === "number") {
    return stats[MEAN_KEY];
  }
  const values = Object.values(stats);
  return values.length > 0 ? values[0] : 0;
}

/**
 * A single labeled activation comparison row with a proportional bar.
 *
 * @param props - The agent label, color, value, and the shared max for scaling.
 * @returns The rendered stat row.
 */
function StatRow({
  label,
  color,
  value,
  max,
}: {
  label: string;
  color: string;
  value: number;
  max: number;
}): JSX.Element {
  const widthPct = max > 0 ? Math.min(100, (value / max) * 100) : 0;

  return (
    <div className="flex items-center gap-3">
      <span
        className="w-16 shrink-0 font-display text-[11px] font-semibold uppercase tracking-wide"
        style={{ color }}
      >
        {label}
      </span>
      <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-argus-black ring-1 ring-inset ring-argus-border">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
          initial={{ width: "0%" }}
          animate={{ width: `${widthPct}%` }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        />
      </div>
      <span className="w-16 shrink-0 text-right font-mono tabular-nums text-xs text-white">
        {value.toFixed(3)}
      </span>
    </div>
  );
}

/**
 * Renders the contested-region disagreement map and per-agent activation stats.
 *
 * @param props - The attention result or null.
 * @returns The rendered map, or a faint placeholder when no result is present.
 */
export default function DisagreementMap({
  attention,
}: DisagreementMapProps): JSX.Element {
  if (attention === null) {
    return (
      <div className="flex flex-col gap-2 rounded-xl border border-dashed border-argus-border bg-argus-surface/40 p-5">
        <span className="font-display text-xs uppercase tracking-[0.2em] text-argus-muted/60">
          Contested region
        </span>
        <p className="font-mono text-xs text-argus-muted/60">
          Awaiting spatial attention analysis…
        </p>
      </div>
    );
  }

  const meanA = regionMean(attention.region_stats_a);
  const meanB = regionMean(attention.region_stats_b);
  const max = Math.max(meanA, meanB);

  return (
    <motion.div
      className="flex flex-col gap-4 rounded-xl border border-argus-border bg-argus-surface p-5"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
    >
      <span className="font-display text-xs font-semibold uppercase tracking-[0.2em] text-argus-danger">
        Contested region
      </span>

      <div className="mx-auto w-full max-w-xs">
        <HeatmapCanvas
          originalImageSrc={`data:image/png;base64,${attention.disagreement_b64}`}
          heatmapB64={null}
          bbox={attention.bbox}
        />
      </div>

      <div className="flex flex-col gap-2.5">
        <span className="font-display text-[11px] uppercase tracking-wide text-argus-muted">
          Mean activation in contested region
        </span>
        <StatRow
          label="Agent A"
          color={AGENT_A_COLOR}
          value={meanA}
          max={max}
        />
        <StatRow
          label="Agent B"
          color={AGENT_B_COLOR}
          value={meanB}
          max={max}
        />
      </div>
    </motion.div>
  );
}
