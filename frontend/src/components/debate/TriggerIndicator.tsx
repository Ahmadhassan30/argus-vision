"use client";

/**
 * TriggerIndicator visualizes the outcome of the debate-trigger evaluation.
 *
 *   - While the trigger is still being evaluated (null) it shows a small
 *     orbital spinner with an "Evaluating disagreement..." caption.
 *   - When the trigger fired, it shows a red "DEBATE TRIGGERED" badge, a
 *     Jensen-Shannon divergence gauge (0..1 scale with the measured value and
 *     the firing threshold marked), and the per-agent prediction entropies.
 *   - When the trigger did not fire, it shows a green "AGENTS AGREE" badge and
 *     a note that the pipeline takes the fast path to consensus.
 */

import { motion } from "framer-motion";
import clsx from "clsx";
import type { TriggerResult } from "@/types/debate";
import LoadingOrbit from "@/components/ui/LoadingOrbit";

/** Props for {@link TriggerIndicator}. */
export interface TriggerIndicatorProps {
  /** The trigger evaluation result, or null while it is being computed. */
  trigger: TriggerResult | null;
}

/**
 * Clamps a value into the [0, 1] range and returns it as a CSS percentage
 * string suitable for absolute positioning along the gauge track.
 *
 * @param value - The raw value to clamp and convert.
 * @returns A percentage string (e.g. "42%").
 */
function toTrackPercent(value: number): string {
  const clamped = Math.min(1, Math.max(0, value));
  return `${clamped * 100}%`;
}

/**
 * The Jensen-Shannon divergence gauge: a horizontal 0..1 track with a marker
 * at the measured divergence and a tick at the firing threshold.
 *
 * @param props - The measured divergence and the firing threshold.
 * @returns The rendered gauge.
 */
function JsGauge({
  jsDivergence,
  thresholdJs,
}: {
  jsDivergence: number;
  thresholdJs: number;
}): JSX.Element {
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between text-[11px]">
        <span className="font-display uppercase tracking-wide text-argus-muted">
          JS divergence
        </span>
        <span className="font-mono tabular-nums text-white">
          {jsDivergence.toFixed(3)}
        </span>
      </div>
      <div className="relative h-2.5 w-full rounded-full bg-argus-black ring-1 ring-inset ring-argus-border">
        {/* Measured divergence fill. */}
        <motion.div
          className="absolute left-0 top-0 h-full rounded-full bg-argus-danger"
          initial={{ width: "0%" }}
          animate={{ width: toTrackPercent(jsDivergence) }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        />
        {/* Firing threshold tick. */}
        <div
          className="absolute top-1/2 h-4 w-0.5 -translate-y-1/2 bg-argus-warning"
          style={{ left: toTrackPercent(thresholdJs) }}
          title={`Threshold ${thresholdJs.toFixed(3)}`}
        />
        {/* Measured-value marker. */}
        <motion.div
          className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white bg-argus-danger"
          initial={{ left: "0%" }}
          animate={{ left: toTrackPercent(jsDivergence) }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        />
      </div>
      <div className="flex items-center justify-between text-[10px] font-mono text-argus-muted">
        <span>0.0</span>
        <span>threshold {thresholdJs.toFixed(2)}</span>
        <span>1.0</span>
      </div>
    </div>
  );
}

/**
 * The debate-trigger status indicator.
 *
 * @param props - The trigger result or null.
 * @returns The rendered indicator.
 */
export default function TriggerIndicator({
  trigger,
}: TriggerIndicatorProps): JSX.Element {
  if (trigger === null) {
    return (
      <div className="flex items-center gap-3 rounded-xl border border-argus-border bg-argus-surface px-4 py-3">
        <LoadingOrbit size={24} />
        <span className="font-display text-sm text-argus-muted">
          Evaluating disagreement...
        </span>
      </div>
    );
  }

  return (
    <motion.div
      className="flex flex-col gap-4 rounded-xl border border-argus-border bg-argus-surface p-5"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
    >
      {trigger.fired ? (
        <>
          <span
            className={clsx(
              "inline-flex w-fit items-center gap-2 rounded-full border px-3 py-1",
              "border-argus-danger bg-argus-danger/10 font-display text-xs font-semibold",
              "uppercase tracking-wide text-argus-danger"
            )}
          >
            <span className="h-2 w-2 rounded-full bg-argus-danger" />
            Debate triggered
          </span>

          <JsGauge
            jsDivergence={trigger.js_divergence}
            thresholdJs={trigger.threshold_js}
          />

          <div className="grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-0.5 rounded-md border border-argus-border bg-argus-black px-3 py-2">
              <span className="font-display text-[11px] uppercase tracking-wide text-argus-agent-a">
                Entropy A
              </span>
              <span className="font-mono tabular-nums text-sm text-white">
                {trigger.entropy_a.toFixed(3)}
              </span>
            </div>
            <div className="flex flex-col gap-0.5 rounded-md border border-argus-border bg-argus-black px-3 py-2">
              <span className="font-display text-[11px] uppercase tracking-wide text-argus-agent-b">
                Entropy B
              </span>
              <span className="font-mono tabular-nums text-sm text-white">
                {trigger.entropy_b.toFixed(3)}
              </span>
            </div>
          </div>

          <p className="font-mono text-[11px] text-argus-muted">
            Entropy threshold {trigger.threshold_entropy.toFixed(2)}
          </p>
        </>
      ) : (
        <>
          <span
            className={clsx(
              "inline-flex w-fit items-center gap-2 rounded-full border px-3 py-1",
              "border-argus-consensus bg-argus-consensus/10 font-display text-xs",
              "font-semibold uppercase tracking-wide text-argus-consensus"
            )}
          >
            <span className="h-2 w-2 rounded-full bg-argus-consensus" />
            Agents agree
          </span>
          <p className="font-display text-sm text-argus-muted">
            Fast path to consensus
          </p>
        </>
      )}
    </motion.div>
  );
}
