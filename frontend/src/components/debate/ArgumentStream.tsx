"use client";

/**
 * ArgumentStream renders one round of the adversarial debate as two side-by-side
 * monospace panels — Agent A on the left, Agent B on the right. Each panel shows
 * that agent's argument text. While an agent is actively streaming, its panel
 * displays the in-progress `streamingText` with a blinking cursor appended;
 * once an argument is complete it shows a subtle checkmark.
 */

import { motion } from "framer-motion";
import clsx from "clsx";
import { AGENT_A_COLOR, AGENT_B_COLOR } from "@/lib/constants";

/** Props for {@link ArgumentStream}. */
export interface ArgumentStreamProps {
  /** Which debate round this stream represents. */
  round: 1 | 2;
  /** Agent A's full argument text (empty until it has produced one). */
  argumentA: string;
  /** Agent B's full argument text (empty until it has produced one). */
  argumentB: string;
  /** Which agent is currently streaming, or null if none. */
  streamingAgent: "A" | "B" | null;
  /** The partial text accumulated so far for the streaming agent. */
  streamingText: string;
}

/** Resolved per-agent display state for one panel. */
interface PanelState {
  /** The text to render (either streaming partial or finished argument). */
  text: string;
  /** Whether this panel is the one currently streaming. */
  isStreaming: boolean;
  /** Whether this panel holds a completed, non-streaming argument. */
  isComplete: boolean;
}

/**
 * A single agent argument panel.
 *
 * @param props - The agent label, color, accent border class, and panel state.
 * @returns The rendered panel.
 */
function AgentPanel({
  label,
  color,
  borderClass,
  state,
}: {
  label: string;
  color: string;
  borderClass: string;
  state: PanelState;
}): JSX.Element {
  return (
    <div
      className={clsx(
        "flex flex-1 flex-col gap-2 rounded-md border-l-4 bg-argus-black",
        "border border-argus-border p-4",
        borderClass
      )}
    >
      <div className="flex items-center justify-between">
        <span
          className="font-display text-xs font-semibold uppercase tracking-wide"
          style={{ color }}
        >
          {label}
        </span>
        {state.isComplete && (
          <span
            className="font-mono text-xs text-argus-consensus"
            title="Argument complete"
            aria-label="Argument complete"
          >
            &#10003;
          </span>
        )}
      </div>
      <p className="whitespace-pre-wrap break-words font-mono text-sm leading-relaxed text-white">
        {state.text}
        {state.isStreaming && (
          <span className="blink-cursor text-argus-muted" aria-hidden="true">
            &#9608;
          </span>
        )}
      </p>
    </div>
  );
}

/**
 * Renders one round of the debate as two monospace argument panels.
 *
 * @param props - Round number, both arguments, and the active stream state.
 * @returns The rendered round.
 */
export default function ArgumentStream({
  round,
  argumentA,
  argumentB,
  streamingAgent,
  streamingText,
}: ArgumentStreamProps): JSX.Element {
  const header =
    round === 1
      ? "ROUND 1 — INITIAL ARGUMENTS"
      : "ROUND 2 — REBUTTALS";

  const stateA: PanelState = {
    text: streamingAgent === "A" ? streamingText : argumentA,
    isStreaming: streamingAgent === "A",
    isComplete: streamingAgent !== "A" && argumentA.length > 0,
  };
  const stateB: PanelState = {
    text: streamingAgent === "B" ? streamingText : argumentB,
    isStreaming: streamingAgent === "B",
    isComplete: streamingAgent !== "B" && argumentB.length > 0,
  };

  return (
    <motion.div
      className="flex flex-col gap-3"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
    >
      <h4 className="font-mono text-xs uppercase tracking-[0.2em] text-argus-muted">
        {header}
      </h4>
      <div className="flex flex-col gap-3 md:flex-row">
        <AgentPanel
          label="Agent A"
          color={AGENT_A_COLOR}
          borderClass="border-l-argus-agent-a"
          state={stateA}
        />
        <AgentPanel
          label="Agent B"
          color={AGENT_B_COLOR}
          borderClass="border-l-argus-agent-b"
          state={stateB}
        />
      </div>
    </motion.div>
  );
}
