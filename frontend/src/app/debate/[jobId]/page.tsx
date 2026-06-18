"use client";

import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useDebateStream } from "@/hooks/useDebateStream";
import AgentCard from "@/components/debate/AgentCard";
import TriggerIndicator from "@/components/debate/TriggerIndicator";
import DisagreementMap from "@/components/debate/DisagreementMap";
import ArgumentStream from "@/components/debate/ArgumentStream";
import ConsensusVerdict from "@/components/debate/ConsensusVerdict";
import LoadingOrbit from "@/components/ui/LoadingOrbit";
import { AGENT_A_COLOR, AGENT_B_COLOR } from "@/lib/constants";
import type { JobStatus } from "@/types/debate";

/** Props injected by the Next.js dynamic route segment. */
interface DebatePageProps {
  params: { jobId: string };
}

/** Terminal job statuses after which the debate is no longer in progress. */
const TERMINAL_STATUSES: ReadonlyArray<JobStatus> = [
  "consensus_done",
  "failed",
];

/** Human-readable labels for each job status shown in the status chip. */
const STATUS_LABELS: Record<JobStatus, string> = {
  queued: "Queued",
  running: "Running agents",
  agents_done: "Agents done",
  trigger_evaluated: "Trigger evaluated",
  attention_computed: "Attention computed",
  debate_round_1: "Debate round 1",
  debate_round_2: "Debate round 2",
  consensus_done: "Consensus reached",
  failed: "Failed",
};

/**
 * Formats an elapsed duration in seconds as `M:SS`.
 *
 * @param totalSeconds - The elapsed time in whole seconds.
 */
function formatElapsed(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

/**
 * Live debate stream page. Subscribes to the job's WebSocket stream via
 * `useDebateStream` and renders both agents, the trigger evaluation, the
 * spatial disagreement map, the round-by-round argument stream, and the final
 * calibrated consensus.
 *
 * @param props.params - The route params containing the job id.
 */
export default function DebatePage({
  params,
}: DebatePageProps): React.JSX.Element {
  const { jobId } = params;
  const state = useDebateStream(jobId);
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);

  // Drive an elapsed-time counter from mount until a terminal status.
  const isTerminal = TERMINAL_STATUSES.includes(state.status);
  useEffect(() => {
    if (isTerminal) {
      return;
    }
    const intervalId = window.setInterval(() => {
      setElapsedSeconds((prev) => prev + 1);
    }, 1000);
    return () => {
      window.clearInterval(intervalId);
    };
  }, [isTerminal]);

  const streamingAgent = state.streamingTokens?.agent ?? null;
  const streamingRound = state.streamingTokens?.round ?? null;
  const streamingText = state.streamingTokens?.text ?? "";

  const round1 = state.argumentsRound1;
  const showRound1 =
    round1.A.length > 0 ||
    round1.B.length > 0 ||
    state.status === "debate_round_1";
  const debateInProgress = !isTerminal;

  return (
    <main className="bg-argus-grid min-h-screen w-full">
      <header className="sticky top-0 z-30 border-b border-argus-border bg-argus-black/85 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-6 py-4">
          <div className="flex items-center gap-4">
            <div className="flex h-10 w-10 items-center justify-center rounded-md border border-argus-border bg-argus-surface font-mono text-xs text-argus-muted">
              IMG
            </div>
            <div className="flex flex-col">
              <span className="font-display text-sm font-bold tracking-[0.3em]">
                ARGUS<span className="text-argus-muted"> VISION</span>
              </span>
              <span className="font-mono text-[11px] text-argus-muted">
                job {jobId}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <span className="rounded-full border border-argus-border bg-argus-surface px-3 py-1 font-mono text-xs text-white">
              {STATUS_LABELS[state.status]}
            </span>
            <span className="font-mono text-xs tabular-nums text-argus-muted">
              {formatElapsed(elapsedSeconds)}
            </span>
          </div>
        </div>
      </header>

      <div className="mx-auto w-full max-w-7xl px-6 py-8">
        {state.error !== null && (
          <div
            role="alert"
            className="mb-8 w-full rounded-lg border border-argus-danger/50 bg-argus-danger/10 px-4 py-3 font-mono text-sm text-argus-danger"
          >
            {state.error}
          </div>
        )}

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <AgentCard
            agentId="A"
            label="AGENT A"
            color={AGENT_A_COLOR}
            result={state.agentA}
            isActive={streamingAgent === "A"}
          />
          <AgentCard
            agentId="B"
            label="AGENT B"
            color={AGENT_B_COLOR}
            result={state.agentB}
            isActive={streamingAgent === "B"}
          />
        </div>

        <div className="mt-8 flex flex-col gap-8">
          <TriggerIndicator trigger={state.trigger} />

          {state.attention !== null && (
            <DisagreementMap attention={state.attention} />
          )}

          {showRound1 && (
            <ArgumentStream
              round={1}
              argumentA={round1.A}
              argumentB={round1.B}
              streamingAgent={streamingRound === 1 ? streamingAgent : null}
              streamingText={streamingRound === 1 ? streamingText : ""}
            />
          )}

          {state.status === "debate_round_2" || state.consensus !== null ? (
            <ArgumentStream
              round={2}
              argumentA={state.argumentsRound2.A}
              argumentB={state.argumentsRound2.B}
              streamingAgent={streamingRound === 2 ? streamingAgent : null}
              streamingText={streamingRound === 2 ? streamingText : ""}
            />
          ) : null}

          <AnimatePresence>
            {state.consensus !== null && (
              <motion.div
                key="consensus"
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 12 }}
                transition={{ duration: 0.5, ease: "easeOut" }}
              >
                <ConsensusVerdict
                  consensus={state.consensus}
                  trigger={state.trigger}
                />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {debateInProgress && (
          <div className="mt-12 flex flex-col items-center justify-center gap-3 py-8">
            <LoadingOrbit size={48} />
            <span className="font-mono text-xs uppercase tracking-widest text-argus-muted">
              {STATUS_LABELS[state.status]}
            </span>
          </div>
        )}
      </div>
    </main>
  );
}
