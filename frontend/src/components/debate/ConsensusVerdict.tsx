"use client";

/**
 * ConsensusVerdict renders the final, calibrated verdict of the pipeline. It
 * fades/scales in when a consensus result becomes available and displays the
 * predicted lesion's full name (large serif), a risk-colored confidence
 * percentage, the calibration metrics (expected calibration error and the
 * learned temperature), and which path produced the result (adversarial debate
 * vs. the fast agreement path). A muted disclaimer makes clear this is a
 * research prototype, not a clinical tool.
 */

import { AnimatePresence, motion } from "framer-motion";
import type { ConsensusResult, TriggerResult } from "@/types/debate";
import { getClassMeta, getRisk } from "@/lib/constants";

/** Props for {@link ConsensusVerdict}. */
export interface ConsensusVerdictProps {
  /** The final consensus result, or null until it is produced. */
  consensus: ConsensusResult | null;
  /** The trigger result, used to label the path taken (debate vs. fast). */
  trigger: TriggerResult | null;
}

/**
 * The final consensus verdict panel.
 *
 * @param props - The consensus result and the trigger result.
 * @returns The rendered verdict, animating in when consensus is available.
 */
export default function ConsensusVerdict({
  consensus,
  trigger,
}: ConsensusVerdictProps): JSX.Element {
  return (
    <AnimatePresence mode="wait">
      {consensus !== null && (
        <motion.div
          key="consensus"
          className="flex flex-col gap-4 rounded-xl border border-argus-consensus/40 bg-argus-surface p-6"
          initial={{ opacity: 0, scale: 0.96, y: 12 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.96, y: 12 }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        >
          <span className="font-display text-xs font-semibold uppercase tracking-[0.2em] text-argus-consensus">
            Consensus verdict
          </span>

          <ConsensusBody consensus={consensus} trigger={trigger} />

          <p className="font-mono text-[11px] text-argus-muted">
            Research prototype. Not for clinical use.
          </p>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

/**
 * The inner body of the verdict: predicted class, confidence, calibration
 * metrics, and the resolution path.
 *
 * @param props - The (non-null) consensus result and the trigger result.
 * @returns The rendered verdict body.
 */
function ConsensusBody({
  consensus,
  trigger,
}: {
  consensus: ConsensusResult;
  trigger: TriggerResult | null;
}): JSX.Element {
  const meta = getClassMeta(consensus.pred_class);
  const fullName = meta ? meta.fullName : consensus.pred_class;
  const confidenceColor = getRisk(consensus.pred_class);
  const confidencePct = (consensus.confidence * 100).toFixed(1);
  const pathLabel = trigger?.fired
    ? "Resolved via adversarial debate"
    : "Fast path — agents agreed";

  return (
    <>
      <div className="flex flex-col gap-1">
        <h2 className="font-serif text-4xl leading-tight text-white">
          {fullName}
        </h2>
        <div className="flex items-baseline gap-2">
          <span
            className="font-mono text-2xl font-semibold tabular-nums"
            style={{ color: confidenceColor }}
          >
            {confidencePct}%
          </span>
          <span className="font-display text-sm text-argus-muted">
            confidence
          </span>
        </div>
      </div>

      <div className="flex flex-wrap gap-3">
        <span className="inline-flex items-center gap-2 rounded-md border border-argus-border bg-argus-black px-3 py-1.5 font-mono text-xs text-white">
          Calibration Error: {consensus.ece.toFixed(2)}
        </span>
        <span className="inline-flex items-center gap-2 rounded-md border border-argus-border bg-argus-black px-3 py-1.5 font-mono text-xs text-white">
          Temperature: {consensus.temperature.toFixed(2)}
        </span>
      </div>

      <p className="font-display text-sm text-argus-muted">{pathLabel}</p>
    </>
  );
}
