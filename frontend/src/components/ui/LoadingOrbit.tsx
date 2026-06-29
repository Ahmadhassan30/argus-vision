"use client";

/**
 * LoadingOrbit — a lightweight spinner: two dots (Agent A blue, Agent B violet)
 * 180° apart orbit a faint centre via a CSS rotation. Sized via `size`. Uses no
 * animation library; reduced-motion is honoured by the global CSS rule.
 */

import { AGENT_A_COLOR, AGENT_B_COLOR } from "@/lib/constants";

export interface LoadingOrbitProps {
  /** Overall pixel diameter of the orbit area. Defaults to 48. */
  size?: number;
}

export default function LoadingOrbit({ size = 48 }: LoadingOrbitProps): React.JSX.Element {
  const dotSize = Math.max(4, Math.round(size / 6));
  const centerSize = Math.max(3, Math.round(size / 10));

  return (
    <div
      role="status"
      aria-label="Loading"
      className="relative inline-block"
      style={{ width: size, height: size }}
    >
      <span
        className="absolute left-1/2 top-1/2 rounded-full bg-ink-faint/40"
        style={{ width: centerSize, height: centerSize, transform: "translate(-50%, -50%)" }}
      />
      <div
        className="absolute inset-0 animate-spin"
        style={{ animationDuration: "1.6s" }}
      >
        <span
          className="absolute left-1/2 top-0 rounded-full"
          style={{ width: dotSize, height: dotSize, backgroundColor: AGENT_A_COLOR, transform: "translate(-50%, 0)" }}
        />
        <span
          className="absolute bottom-0 left-1/2 rounded-full"
          style={{ width: dotSize, height: dotSize, backgroundColor: AGENT_B_COLOR, transform: "translate(-50%, 0)" }}
        />
      </div>
    </div>
  );
}
