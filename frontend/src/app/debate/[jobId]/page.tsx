"use client";

/**
 * The live debate page — Argus Vision's centrepiece. It subscribes to the job's
 * WebSocket stream, orchestrates client-side argument generation, and stages the
 * whole thing as a cinematic experience: an ambient WebGL field that reacts to
 * the debate, a procedure timeline, two arguing agents, the divergence gate, the
 * spatial-attention grid, and the calibrated consensus verdict.
 */

import { useEffect, useMemo, useRef, useState } from "react";

import { useDebateStream } from "@/hooks/useDebateStream";
import { useArgumentStream } from "@/hooks/useArgumentStream";
import { loadJobImage } from "@/lib/sessionImage";
import type { UiPhase } from "@/lib/debateReducer";
import type { FieldMode } from "@/lib/webgl-particles";

import WebGLBackground from "@/components/debate/WebGLBackground";
import TimelineRail from "@/components/debate/TimelineRail";
import AgentCard from "@/components/debate/AgentCard";
import VSDivider from "@/components/debate/VSDivider";
import TriggerPanel from "@/components/debate/TriggerPanel";
import DisagreementMap from "@/components/debate/DisagreementMap";
import ConsensusVerdict from "@/components/debate/ConsensusVerdict";

interface DebatePageProps {
  params: { jobId: string };
}

interface StatusPill {
  label: string;
  color: string;
  tint: string;
  pulse: "none" | "slow" | "fast" | "check";
}

/** Map the coarse UI phase to a header status pill. */
function statusPill(phase: UiPhase): StatusPill {
  switch (phase) {
    case "running":
    case "agents_done":
      return { label: "Agents running", color: "#2563EB", tint: "rgba(37,99,235,0.12)", pulse: "slow" };
    case "triggered":
    case "attention":
    case "consensus":
      return { label: "Debate triggered", color: "#D97706", tint: "rgba(217,119,6,0.12)", pulse: "fast" };
    case "fast_path":
      return { label: "Fast path", color: "#059669", tint: "rgba(5,150,105,0.12)", pulse: "slow" };
    case "resolved":
      return { label: "Resolved", color: "#059669", tint: "rgba(5,150,105,0.12)", pulse: "check" };
    case "error":
      return { label: "Error", color: "#DC2626", tint: "rgba(220,38,38,0.12)", pulse: "none" };
    default:
      return { label: "Awaiting", color: "#94A3B8", tint: "rgba(148,163,184,0.12)", pulse: "none" };
  }
}

/** Map the phase to the WebGL field behaviour. */
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

export default function DebatePage({ params }: DebatePageProps): React.JSX.Element {
  const { jobId } = params;
  const state = useDebateStream(jobId);
  const args = useArgumentStream(state, jobId);

  // Source lesion image handed over from the upload page (client-only).
  const [sourceImage, setSourceImage] = useState<string | null>(null);
  useEffect(() => {
    setSourceImage(loadJobImage(jobId));
  }, [jobId]);

  // Elapsed timer until a terminal phase.
  const [elapsed, setElapsed] = useState(0);
  const terminal = state.phase === "resolved" || state.phase === "error";
  useEffect(() => {
    if (terminal) return;
    const id = window.setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => window.clearInterval(id);
  }, [terminal]);

  // Amber alert banner when the trigger fires (auto-dismiss).
  const [showAlert, setShowAlert] = useState(false);
  useEffect(() => {
    if (state.triggerFired !== true) return;
    setShowAlert(true);
    const id = window.setTimeout(() => setShowAlert(false), 3500);
    return () => window.clearTimeout(id);
  }, [state.triggerFired]);

  // One-shot green sweep across the arena on resolution.
  const [sweep, setSweep] = useState(false);
  const sweptRef = useRef(false);
  useEffect(() => {
    if (state.consensus !== null && !sweptRef.current) {
      sweptRef.current = true;
      setSweep(true);
      const id = window.setTimeout(() => setSweep(false), 1000);
      return () => window.clearTimeout(id);
    }
    return undefined;
  }, [state.consensus]);

  const pill = useMemo(() => statusPill(state.phase), [state.phase]);
  const mode = useMemo(() => fieldMode(state.phase), [state.phase]);

  const hasAgents = state.agentA !== null && state.agentB !== null;
  // The spatial panel is expected only on the debate path.
  const awaitingAttention =
    state.triggerFired === true && state.attention === null && state.consensus === null;

  return (
    <main className="bg-theatre relative min-h-screen w-full">
      <WebGLBackground mode={mode} />

      {/* Header */}
      <header className="sticky top-0 z-20 border-b border-hairline bg-surface/85 backdrop-blur-md shadow-panel">
        <div className="mx-auto flex h-16 max-w-[1400px] items-center justify-between gap-4 px-6">
          <div className="flex items-center gap-3">
            <span className="grid h-9 w-9 place-items-center rounded-md border border-hairline bg-surface font-mono text-[10px] text-ink-faint">
              AV
            </span>
            <div className="leading-tight">
              <div className="font-display text-xl text-ink">Argus</div>
              <div className="-mt-1 text-[10px] font-medium uppercase tracking-[0.2em] text-ink-faint">
                Vision
              </div>
            </div>
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
        {state.error !== null && (
          <div
            role="alert"
            className="mb-6 rounded-xl border border-danger/40 bg-danger/5 px-4 py-3 font-mono text-sm text-danger"
          >
            {state.error}
          </div>
        )}

        <div className="grid grid-cols-1 gap-8 lg:grid-cols-[280px_1fr]">
          {/* Left rail */}
          <aside className="lg:sticky lg:top-24 lg:self-start">
            <TimelineRail
              completedAt={state.completedAt}
              phase={state.phase}
              triggerFired={state.triggerFired}
            />
          </aside>

          {/* Main column */}
          <div className="flex flex-col gap-6">
            {sourceImage && (
              <section className="flex items-center gap-4 rounded-2xl border border-hairline bg-surface p-4 shadow-panel">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={sourceImage}
                  alt="Dermoscopic specimen under analysis"
                  className="h-20 w-20 shrink-0 rounded-xl border border-hairline object-cover"
                />
                <div>
                  <div className="text-[11px] font-medium uppercase tracking-[0.12em] text-ink-faint">
                    Specimen
                  </div>
                  <div className="font-display text-lg text-ink">Dermoscopic lesion</div>
                  <div className="text-xs text-ink-soft">
                    Two agents are analyzing this image in real time.
                  </div>
                </div>
              </section>
            )}

            {/* Arena */}
            <div className="relative">
              {showAlert && (
                <div className="mb-4 flex items-center gap-2 rounded-xl border border-warning/40 bg-warning/10 px-4 py-2.5 text-sm text-warning animate-panel-enter">
                  <span className="h-2 w-2 rounded-full bg-warning animate-pulse-dot-fast" />
                  Diagnostic disagreement detected — initiating spatial debate protocol.
                </div>
              )}

              {sweep && (
                <div className="pointer-events-none absolute inset-0 z-20 overflow-hidden rounded-2xl">
                  <div
                    className="absolute inset-y-0 w-1/3 animate-sweep"
                    style={{
                      background:
                        "linear-gradient(90deg, transparent, rgba(5,150,105,0.22), transparent)",
                    }}
                  />
                </div>
              )}

              <div className="flex flex-col gap-4 lg:grid lg:grid-cols-[1fr_auto_1fr] lg:items-stretch lg:gap-4">
                <AgentCard
                  agentId="A"
                  result={state.agentA}
                  round1={args.round1.A}
                  rebuttal={args.rebuttal.A}
                  argumentActive={args.agentActive.A}
                  thinking={state.agentA === null}
                />
                <div className="lg:hidden">
                  <VSDivider orientation="horizontal" triggerFired={state.triggerFired} />
                </div>
                <div className="hidden lg:block">
                  <VSDivider orientation="vertical" triggerFired={state.triggerFired} />
                </div>
                <AgentCard
                  agentId="B"
                  result={state.agentB}
                  round1={args.round1.B}
                  rebuttal={args.rebuttal.B}
                  argumentActive={args.agentActive.B}
                  thinking={state.agentB === null}
                />
              </div>
            </div>

            {/* Divergence gate — appears once both agents have classified. */}
            {hasAgents && <TriggerPanel trigger={state.trigger} />}

            {/* Spatial attention */}
            {state.attention !== null ? (
              <DisagreementMap attention={state.attention} sourceB64={sourceImage} />
            ) : awaitingAttention ? (
              <section className="rounded-2xl border border-hairline bg-surface p-6 shadow-panel">
                <h3 className="font-display text-xl text-ink">Spatial attention analysis</h3>
                <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
                  {[0, 1, 2, 3].map((i) => (
                    <div key={i} className="shimmer animate-shimmer aspect-square w-full rounded-lg" />
                  ))}
                </div>
                <p className="mt-4 text-xs text-ink-faint">
                  Spatial analysis in progress — computing Grad-CAM++ and attention rollout…
                </p>
              </section>
            ) : null}

            {/* Consensus verdict */}
            {state.consensus !== null && (
              <ConsensusVerdict
                consensus={state.consensus}
                trigger={state.trigger}
                synthesis={args.synthesis}
                synthesisActive={args.streaming.synthesis}
              />
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
