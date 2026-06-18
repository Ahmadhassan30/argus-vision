"use client";

/**
 * LoadingOrbit is a lightweight loading indicator: two dots (one in the Agent A
 * color and one in the Agent B color) positioned 180deg apart orbit a faint
 * center dot via an infinite linear rotation. It is sized via the optional
 * `size` prop and exposes an accessible "loading" label.
 */

import { motion } from "framer-motion";
import { AGENT_A_COLOR, AGENT_B_COLOR } from "@/lib/constants";

/** Props for {@link LoadingOrbit}. */
export interface LoadingOrbitProps {
  /** Overall pixel diameter of the orbit area. Defaults to 48. */
  size?: number;
}

/**
 * A two-dot orbital spinner indicating an in-progress operation.
 *
 * @param props - Optional `size` (pixel diameter, default 48).
 * @returns The rendered loading indicator.
 */
export default function LoadingOrbit({
  size = 48,
}: LoadingOrbitProps): JSX.Element {
  const dotSize = Math.max(4, Math.round(size / 6));
  const centerSize = Math.max(3, Math.round(size / 10));

  return (
    <div
      role="status"
      aria-label="loading"
      className="relative inline-block"
      style={{ width: size, height: size }}
    >
      {/* Faint center dot. */}
      <span
        className="absolute left-1/2 top-1/2 rounded-full bg-argus-muted/40"
        style={{
          width: centerSize,
          height: centerSize,
          transform: "translate(-50%, -50%)",
        }}
      />

      {/* Rotating wrapper carrying the two orbiting dots. */}
      <motion.div
        className="absolute inset-0"
        animate={{ rotate: 360 }}
        transition={{ duration: 1.6, ease: "linear", repeat: Infinity }}
      >
        {/* Agent A dot at the top (0deg). */}
        <span
          className="absolute left-1/2 top-0 rounded-full"
          style={{
            width: dotSize,
            height: dotSize,
            backgroundColor: AGENT_A_COLOR,
            transform: "translate(-50%, 0)",
          }}
        />
        {/* Agent B dot at the bottom (180deg apart). */}
        <span
          className="absolute bottom-0 left-1/2 rounded-full"
          style={{
            width: dotSize,
            height: dotSize,
            backgroundColor: AGENT_B_COLOR,
            transform: "translate(-50%, 0)",
          }}
        />
      </motion.div>
    </div>
  );
}
