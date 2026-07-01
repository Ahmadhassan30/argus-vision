"use client";

/**
 * The live debate page — Argus Vision's centrepiece. It subscribes to the job's
 * WebSocket stream and runs a real turn-taking debate engine on top of the
 * classifier's outputs: two agents argue back and forth, revising their beliefs
 * each turn until they converge, then the calibrated consensus is revealed. An
 * ambient WebGL field, a procedure timeline, the divergence gate and the spatial
 * attention grid frame the conversation.
 */

import { useEffect, useMemo, useRef, useState } from "react";

import { useDebateStream } from "@/hooks/useDebateStream";
import { useDebateEngine } from "@/hooks/useDebateEngine";
import { loadJobImage } from "@/lib/sessionImage";
import { getClassName } from "@/lib/constants";
import type { UiPhase } from "@/lib/debateReducer";
import type { FieldMode } from "@/lib/webgl-particles";

import WebGLBackground from "@/components/debate/WebGLBackground";
import TimelineRail from "@/components/debate/TimelineRail";
import AgentScoreboard, { type AgentStatus } from "@/components/debate/AgentScoreboard";
import VSDivider from "@/components/debate/VSDivider";
import DebateTranscript from "@/components/debate/DebateTranscript";
import TriggerPanel from "@/components/debate/TriggerPanel";
import DisagreementMap from "@/components/debate/DisagreementMap";
import ConsensusVerdict from "@/components/debate/ConsensusVerdict";

interface DebatePageProps {
  params: { jobId: string };
}

function fieldMode(phase: UiPhase): FieldMode {
  if (phase === "resolved") return "consensus";
  if (phase === "triggered" || phase === "attention" || phase === "consensus") return "debate";
  return "idle";
}

function formatElapsed(totalSeconds: number): string {
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function leadClass(probs: Record<string, number> | null): string | null {
  if (!probs) return null;
  let best: string | null = null;
  let bv = -Infinity;
  for (const [k, v] of Object.entries(probs)) {
    if (v > bv) {
      bv = v;
      best = k;
    }
  }
  return best;
}

export default function DebatePage({ params }: DebatePageProps): React.JSX.Element {
  const { jobId } = params;
  const ws = useDebateStream(jobId);
  const debate = useDebateEngine(ws, jobId);

  const [sourceImage, setSourceImage] = useState<string | null>(null);
  useEffect(() => {
    setSourceImage(loadJobImage(jobId));
  }, [jobId]);

  const debateRunning = debate.active && !debate.finished;
  const showConsensus = debate.finished && ws.consensus !== null;

  // Elapsed timer until resolution.
  const [elapsed, setElapsed] = useState(0);
  const terminal = showConsensus || ws.phase === "error";
  useEffect(() => {
    if (terminal) return;
    const id = window.setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => window.clearInterval(id);
  }, [terminal]);

  // Amber alert when the trigger fires.
  const [showAlert, setShowAlert] = useState(false);
  useEffect(() => {
    if (ws.triggerFired !== true) return;
    setShowAlert(true);
    const id = window.setTimeout(() => setShowAlert(false), 3500);
    return () => window.clearTimeout(id);
  }, [ws.triggerFired]);

  // Green sweep on resolution.
  const [sweep, setSweep] = useState(false);
  const sweptRef = useRef(false);
  useEffect(() => {
    if (showConsensus && !sweptRef.current) {
      sweptRef.current = true;
      setSweep(true);
      const id = window.setTimeout(() => setSweep(false), 1000);
      return () => window.clearTimeout(id);
    }
    return undefined;
  }, [showConsensus]);

  const mode: FieldMode = showConsensus ? "consensus" : debateRunning ? "debate" : fieldMode(ws.phase);

  // Header status pill.
  const pill = useMemo(() => {
    if (ws.phase === "error") return { label: "Error", color: "#DC2626", tint: "rgba(220,38,38,0.12)", pulse: "none" as const };
    if (showConsensus) return { label: "Resolved", color: "#059669", tint: "rgba(5,150,105,0.12)", pulse: "check" as const };
    if (debateRunning) return { label: `Debating · R${debate.round || 1}`, color: "#7C3AED", tint: "rgba(124,58,237,0.12)", pulse: "fast" as const };
    if (ws.agentA && ws.agentB) return { label: "Agents ready", color: "#2563EB", tint: "rgba(37,99,235,0.12)", pulse: "slow" as const };
    if (ws.phase === "running") return { label: "Agents running", color: "#2563EB", tint: "rgba(37,99,235,0.12)", pulse: "slow" as const };
    return { label: "Awaiting", color: "#94A3B8", tint: "rgba(148,163,184,0.12)", pulse: "none" as const };
  }, [ws.phase, ws.agentA, ws.agentB, debateRunning, debate.round, showConsensus]);

  // Per-agent scoreboard inputs (live belief while debating, else classifier).
  const aProbs = debate.turns.length > 0 ? debate.beliefA : ws.agentA?.result.probabilities ?? null;
  const bProbs = debate.turns.length > 0 ? debate.beliefB : ws.agentB?.result.probabilities ?? null;
  const aConf = debate.turns.length > 0 ? debate.confA : ws.agentA?.result.confidence ?? 0;
  const bConf = debate.turns.length > 0 ? debate.confB : ws.agentB?.result.confidence ?? 0;

  const statusFor = (agent: "A" | "B", has: boolean): AgentStatus => {
    if (!has) return "thinking";
    if (debate.finished) return "settled";
    if (debateRunning && debate.speaker === agent) return "speaking";
    if (debateRunning) return "listening";
    return "thinking";
  };

  const convergedClass = debate.finished && debate.converged ? leadClass(debate.beliefA) : null;
  const recap = ws.consensus
    ? `After ${debate.round} rounds, the agents reconciled their reading to ${getClassName(ws.consensus.pred_class)}; the calibrated head finalizes it at ${(ws.consensus.confidence * 100).toFixed(0)}% confidence.`
    : "";

  return (
    <main className="bg-theatre relative min-h-screen w-full">
      <WebGLBackground mode={mode} />

      {/* Header */}
      <header className="sticky top-0 z-20 border-b border-hairline bg-surface/85 backdrop-blur-md shadow-panel">
        <div className="mx-auto flex h-16 max-w-[1400px] items-center justify-between gap-4 px-6">
          <div className="flex items-center gap-3">
            <img
              src="/logo.png"
              alt="Argus Vision Logo"
              className="h-12 w-auto object-contain"
            />
            <span className="ml-2 hidden rounded-full border border-hairline px-2.5 py-1 font-mono text-[11px] text-ink-faint sm:inline">
              JOB · {jobId.slice(0, 8)}
            </span>
          </div>
          <div className="flex items-center gap-3">
            <span
              className="inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-[11px] font-semibold uppercase tracking-[0.12em]"
              style={{ backgroundColor: pill.tint, color: pill.color }}
            >
              {pill.pulse === "check" ? (
                <svg viewBox="0 0 24 24" className="h-3 w-3" fill="none" aria-hidden>
                  <path d="M5 13l4 4L19 7" stroke={pill.color} strokeWidth={3} strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              ) : (
                <span
                  className={[
                    "h-2 w-2 rounded-full",
                    pill.pulse === "slow" ? "animate-pulse-dot" : "",
                    pill.pulse === "fast" ? "animate-pulse-dot-fast" : "",
                  ].join(" ")}
                  style={{ backgroundColor: pill.color }}
                />
              )}
              {pill.label}
            </span>
            <span className="font-mono text-xs tabular text-ink-faint">{formatElapsed(elapsed)}</span>
          </div>
        </div>
      </header>

      {/* Body */}
      <div className="relative z-10 mx-auto w-full max-w-[1400px] px-6 py-8">
        {ws.error !== null && (
          <div role="alert" className="mb-6 rounded-xl border border-danger/40 bg-danger/5 px-4 py-3 font-mono text-sm text-danger">
            {ws.error}
          </div>
        )}

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[260px_1fr]">
          <aside className="lg:sticky lg:top-24 lg:self-start">
            <TimelineRail completedAt={ws.completedAt} phase={ws.phase} triggerFired={ws.triggerFired} />
          </aside>

          <div className="flex flex-col gap-5">
            {sourceImage && (
              <section className="flex items-center gap-3 rounded-2xl border border-hairline bg-surface px-4 py-3 shadow-panel">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={sourceImage} alt="Dermoscopic specimen under analysis" className="h-14 w-14 shrink-0 rounded-lg border border-hairline object-cover" />
                <div className="leading-tight">
                  <div className="text-[10px] font-medium uppercase tracking-[0.12em] text-ink-faint">Specimen</div>
                  <div className="font-display text-lg text-ink">Dermoscopic lesion</div>
                </div>
              </section>
            )}

            {/* Scoreboards (live evolving beliefs) */}
            <div className="relative">
              {showAlert && (
                <div className="mb-4 flex items-center gap-2 rounded-xl border border-warning/40 bg-warning/10 px-4 py-2.5 text-sm text-warning animate-panel-enter">
                  <span className="h-2 w-2 rounded-full bg-warning animate-pulse-dot-fast" />
                  Diagnostic disagreement detected — opening the debate.
                </div>
              )}
              {sweep && (
                <div className="pointer-events-none absolute inset-0 z-20 overflow-hidden rounded-2xl">
                  <div className="absolute inset-y-0 w-1/3 animate-sweep" style={{ background: "linear-gradient(90deg, transparent, rgba(5,150,105,0.22), transparent)" }} />
                </div>
              )}
              <div className="flex flex-col gap-4 lg:grid lg:grid-cols-[1fr_auto_1fr] lg:items-stretch lg:gap-4">
                <AgentScoreboard agentId="A" probs={aProbs} confidence={aConf} topClass={leadClass(aProbs)} status={statusFor("A", ws.agentA !== null)} />
                <div className="lg:hidden">
                  <VSDivider orientation="horizontal" triggerFired={ws.triggerFired} />
                </div>
                <div className="hidden lg:block">
                  <VSDivider orientation="vertical" triggerFired={ws.triggerFired} />
                </div>
                <AgentScoreboard agentId="B" probs={bProbs} confidence={bConf} topClass={leadClass(bProbs)} status={statusFor("B", ws.agentB !== null)} />
              </div>
            </div>

            {/* The conversation */}
            {(debate.active || debate.turns.length > 0) && (
              <DebateTranscript
                turns={debate.turns}
                agreement={debate.agreement}
                round={debate.round}
                converged={debate.converged}
                finished={debate.finished}
                active={debateRunning}
                convergedClass={convergedClass}
              />
            )}

            {/* Divergence gate */}
            {ws.agentA && ws.agentB && <TriggerPanel trigger={ws.trigger} />}

            {/* Spatial attention */}
            {ws.attention !== null ? (
              <DisagreementMap attention={ws.attention} sourceB64={sourceImage} />
            ) : ws.triggerFired === true && ws.consensus === null ? (
              <section className="rounded-2xl border border-hairline bg-surface p-6 shadow-panel">
                <h3 className="font-display text-xl text-ink">Spatial attention analysis</h3>
                <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
                  {[0, 1, 2, 3].map((i) => (
                    <div key={i} className="shimmer animate-shimmer aspect-square w-full rounded-lg" />
                  ))}
                </div>
                <p className="mt-4 text-xs text-ink-faint">Spatial analysis in progress — computing Grad-CAM++ and attention rollout…</p>
              </section>
            ) : null}

            {/* Consensus — only once the debate has actually resolved. */}
            {showConsensus && ws.consensus && (
              <ConsensusVerdict consensus={ws.consensus} trigger={ws.trigger} synthesis={recap} synthesisActive={false} />
            )}
            {debate.finished && ws.consensus === null && (
              <section className="rounded-2xl border border-hairline bg-surface p-6 text-sm text-ink-faint shadow-panel">
                The agents have settled — finalizing the calibrated verdict…
              </section>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
