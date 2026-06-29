"use client";

/**
 * VSDivider — the hairline separator that stages the two agent cards against one
 * another. A thin rule runs the length of the gutter with a circular "VS" badge
 * floating at its centre. The divider is purely decorative until the divergence
 * trigger fires: at that moment the badge warms to amber, emits a single
 * outward flash, and keeps a soft amber ring to mark that the agents are now in
 * open conflict. Works as a vertical gutter (side-by-side cards) or a horizontal
 * rule (stacked cards below 768px), and all colour shifts ease smoothly.
 */

interface VSDividerProps {
  /** Whether the divergence trigger has fired; null while undecided. */
  triggerFired: boolean | null;
  /** Layout axis of the divider (default "vertical"). */
  orientation?: "vertical" | "horizontal";
}

export default function VSDivider({
  triggerFired,
  orientation = "vertical",
}: VSDividerProps): React.JSX.Element {
  const fired = triggerFired === true;
  const isVertical = orientation === "vertical";

  return (
    <div
      aria-hidden
      className={[
        "relative grid place-items-center",
        isVertical ? "h-full w-11" : "h-11 w-full",
      ].join(" ")}
    >
      {/* The hairline rule running the length of the gutter. */}
      <span
        className={[
          "absolute bg-hairline",
          isVertical
            ? "inset-y-0 left-1/2 w-px -translate-x-1/2"
            : "inset-x-0 top-1/2 h-px -translate-y-1/2",
        ].join(" ")}
      />

      {/* The "VS" badge, centred over the rule. */}
      <span
        className={[
          "relative grid h-11 w-11 place-items-center rounded-full border-2 bg-surface",
          "font-body text-xs font-bold uppercase tracking-[0.1em]",
          "transition-colors duration-300",
          fired
            ? "border-warning text-warning shadow-glow-warning animate-amber-flash"
            : "border-hairline text-ink-faint",
        ].join(" ")}
      >
        VS
      </span>
    </div>
  );
}
