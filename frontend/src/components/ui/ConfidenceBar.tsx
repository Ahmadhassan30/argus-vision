"use client";

/**
 * ConfidenceBar renders a labeled, animated horizontal bar representing a
 * probability/confidence value in the range [0, 1]. The fill animates to its
 * target width and is tinted by the supplied color; the numeric percentage is
 * shown on the right in the monospace font.
 */

import { motion } from "framer-motion";
import clsx from "clsx";

/** Props for {@link ConfidenceBar}. */
export interface ConfidenceBarProps {
  /** Text label shown on the left (e.g. a class code like "MEL"). */
  label: string;
  /** Confidence in the range [0, 1]; rendered as a percentage. */
  value: number;
  /** CSS color string used to tint the animated fill. */
  color: string;
}

/**
 * A single horizontal confidence bar with an animated fill.
 *
 * @param props - The label, value (0..1) and fill color.
 * @returns The rendered confidence bar element.
 */
export default function ConfidenceBar({
  label,
  value,
  color,
}: ConfidenceBarProps): JSX.Element {
  const clamped = Math.min(1, Math.max(0, value));
  const percent = (clamped * 100).toFixed(1);

  return (
    <div className="flex w-full flex-col gap-1">
      <div className="flex items-center justify-between text-xs">
        <span className="font-display text-argus-muted">{label}</span>
        <span className="font-mono tabular-nums text-white">{percent}%</span>
      </div>
      <div
        className={clsx(
          "h-2 w-full overflow-hidden rounded-full bg-argus-surface",
          "ring-1 ring-inset ring-argus-border"
        )}
      >
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
          initial={{ width: "0%" }}
          animate={{ width: `${clamped * 100}%` }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}
