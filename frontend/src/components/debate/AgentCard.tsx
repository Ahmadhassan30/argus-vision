"use client";

/**
 * AgentCard renders a single agent's panel within the debate UI. While the
 * agent is still computing (result === null) it shows an animated skeleton.
 * Once a result is available it shows the predicted class as a {@link ClassBadge}
 * with its confidence, the top-3 class probabilities as {@link ConfidenceBar}s,
 * and a {@link HeatmapCanvas} rendering the agent's saliency heatmap. When the
 * card is the active agent it gains a soft glow in the agent's brand color.
 */

import { motion } from "framer-motion";
import clsx from "clsx";
import type { AgentResult } from "@/types/debate";
import { getClassMeta, type RiskLevel } from "@/lib/constants";
import ClassBadge from "@/components/ui/ClassBadge";
import ConfidenceBar from "@/components/ui/ConfidenceBar";
import HeatmapCanvas from "@/components/debate/HeatmapCanvas";

/** Number of top-probability classes shown as confidence bars. */
const TOP_K = 3;

/** Props for {@link AgentCard}. */
export interface AgentCardProps {
  /** Which agent this card represents ("A" or "B"). */
  agentId: "A" | "B";
  /** Human-readable label (e.g. "Agent A · EfficientNet-B4"). */
  label: string;
  /** The agent's brand color (e.g. argus-agent-a). */
  color: string;
  /** The agent's result, or null while inference is still running. */
  result: AgentResult | null;
  /** Whether this agent is currently the active/spotlighted one. */
  isActive: boolean;
}

/** A single [classId, probability] pair extracted from a probability map. */
interface RankedClass {
  /** The ISIC class identifier (e.g. "MEL"). */
  id: string;
  /** The class probability in [0, 1]. */
  prob: number;
}

/**
 * Returns the top-K classes by probability, sorted descending.
 *
 * @param probabilities - Map of class id to probability.
 * @returns The K highest-probability classes.
 */
function topClasses(probabilities: Record<string, number>): RankedClass[] {
  return Object.entries(probabilities)
    .map(([id, prob]) => ({ id, prob }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, TOP_K);
}

/**
 * The animated loading skeleton shown before a result is available.
 *
 * @returns The skeleton element.
 */
function AgentSkeleton(): JSX.Element {
  return (
    <div className="flex flex-col gap-4" aria-hidden="true">
      <motion.div
        className="h-6 w-24 rounded-full bg-argus-border"
        animate={{ opacity: [0.4, 0.9, 0.4] }}
        transition={{ duration: 1.4, ease: "easeInOut", repeat: Infinity }}
      />
      <div className="flex flex-col gap-3">
        {Array.from({ length: TOP_K }).map((_, index) => (
          <motion.div
            key={index}
            className="h-3 w-full rounded-full bg-argus-border"
            animate={{ opacity: [0.4, 0.9, 0.4] }}
            transition={{
              duration: 1.4,
              ease: "easeInOut",
              repeat: Infinity,
              delay: index * 0.15,
            }}
          />
        ))}
      </div>
      <motion.div
        className="aspect-square w-full rounded-md bg-argus-border"
        animate={{ opacity: [0.35, 0.7, 0.35] }}
        transition={{ duration: 1.6, ease: "easeInOut", repeat: Infinity }}
      />
    </div>
  );
}

/**
 * The per-agent debate card.
 *
 * @param props - Agent identity, color, result, and active state.
 * @returns The rendered agent card.
 */
export default function AgentCard({
  agentId,
  label,
  color,
  result,
  isActive,
}: AgentCardProps): JSX.Element {
  const ranked: RankedClass[] = result
    ? topClasses(result.result.probabilities)
    : [];
  const predMeta = result ? getClassMeta(result.result.pred_class) : undefined;
  const predRisk: RiskLevel = predMeta ? predMeta.risk : "medium";
  const confidencePct = result
    ? (result.result.confidence * 100).toFixed(1)
    : null;

  return (
    <motion.section
      aria-label={label}
      data-agent={agentId}
      className={clsx(
        "flex w-full flex-col gap-4 rounded-xl border bg-argus-surface p-5",
        "border-argus-border"
      )}
      animate={{
        boxShadow: isActive
          ? `0 0 0 1px ${color}, 0 0 24px 2px ${color}66`
          : "0 0 0 0 rgba(0,0,0,0)",
      }}
      transition={{ duration: 0.4, ease: "easeOut" }}
    >
      <header className="flex items-center justify-between">
        <h3
          className="font-display text-sm font-semibold uppercase tracking-wide"
          style={{ color }}
        >
          {label}
        </h3>
        {isActive && (
          <motion.span
            className="h-2.5 w-2.5 rounded-full"
            style={{ backgroundColor: color }}
            animate={{ opacity: [0.4, 1, 0.4] }}
            transition={{ duration: 1.2, ease: "easeInOut", repeat: Infinity }}
            aria-hidden="true"
          />
        )}
      </header>

      {result === null ? (
        <AgentSkeleton />
      ) : (
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between gap-2">
            <ClassBadge className={result.result.pred_class} risk={predRisk} />
            <span className="font-mono text-sm tabular-nums text-white">
              {confidencePct}%
            </span>
          </div>

          <div className="flex flex-col gap-2.5">
            {ranked.map((entry) => (
              <ConfidenceBar
                key={entry.id}
                label={entry.id}
                value={entry.prob}
                color={color}
              />
            ))}
          </div>

          <HeatmapCanvas
            originalImageSrc={null}
            heatmapB64={result.heatmap_b64}
            bbox={null}
          />
        </div>
      )}
    </motion.section>
  );
}
