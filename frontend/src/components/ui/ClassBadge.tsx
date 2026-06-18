"use client";

/**
 * ClassBadge renders an ISIC class code as a color-coded pill. The color is
 * derived from the lesion risk level (low/medium/high) using the shared
 * RISK_COLORS mapping. Note: the text prop is literally named `className` per
 * the project contract, but it carries the ISIC class code (e.g. "MEL"), not a
 * CSS class string.
 */

import clsx from "clsx";
import { RISK_COLORS, type RiskLevel } from "@/lib/constants";

/** Props for {@link ClassBadge}. */
export interface ClassBadgeProps {
  /** The ISIC class code to display (named `className` per the contract). */
  className: string;
  /** Risk level used to color-code the pill. */
  risk: RiskLevel;
}

/**
 * A pill-shaped badge showing an ISIC class code, tinted by risk level.
 *
 * @param props - The class code (`className`) and its `risk` level.
 * @returns The rendered badge element.
 */
export default function ClassBadge({
  className,
  risk,
}: ClassBadgeProps): JSX.Element {
  const color = RISK_COLORS[risk];

  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-full border px-2.5 py-0.5",
        "font-display text-xs font-semibold uppercase tracking-wide"
      )}
      style={{
        color,
        borderColor: color,
        backgroundColor: `${color}1A`,
      }}
      title={`${className} (${risk} risk)`}
    >
      {className}
    </span>
  );
}
