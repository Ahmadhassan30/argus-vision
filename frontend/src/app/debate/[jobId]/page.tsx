"use client";

/**
 * Debate Page — DICOM Workstation Redesign.
 *
 * Implements a high-fidelity replica of the medical imaging software UI (DICOM Viewer)
 * shown in the reference image, preserving all active state logic.
 *
 * Layout Structure:
 * - Header: Zeliha UNLUEL patient identity header replica (Subject/Specimen Info).
 * - Left Panel: Search toolbar (Vertical action icons) + Clinical study explorer.
 * - Center Panel: 2x2 clinical viewports (Coronal, Sagittal, Axial, 3D Reconstruction)
 *   displaying source specimen, Agent A Grad-CAM++, Agent B Attention Rollout, and Disagreement Map.
 * - Right Panel: Stacked diagnostic "Seriler" (series) representing Agent A, Agent B,
 *   Consensus Verdict, and transaction log.
 */

import { useEffect, useMemo, useState } from "react";

import { useDebateStream } from "@/hooks/useDebateStream";
import { useDebateEngine } from "@/hooks/useDebateEngine";
import { loadJobImage } from "@/lib/sessionImage";

import TimelineRail from "@/components/debate/TimelineRail";
import AgentScoreboard, { type AgentStatus } from "@/components/debate/AgentScoreboard";
import DebateTranscript from "@/components/debate/DebateTranscript";
import TriggerPanel from "@/components/debate/TriggerPanel";
import ConsensusVerdict from "@/components/debate/ConsensusVerdict";
import DisagreementMap from "@/components/debate/DisagreementMap";
import HeatmapCanvas from "@/components/debate/HeatmapCanvas";
import { AGENT_A, AGENT_B } from "@/lib/constants";

interface DebatePageProps {
  params: { jobId: string };
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
  const [activeSeries, setActiveSeries] = useState<"A" | "B" | "consensus" | "log">("consensus");
  const [selectedViewport, setSelectedViewport] = useState<1 | 2 | 3 | 4>(1);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    setSourceImage(loadJobImage(jobId));
  }, [jobId]);

  const formattedDate = useMemo(() => {
    if (!mounted) return "";
    return new Date().toLocaleDateString();
  }, [mounted]);

  const formattedTime = useMemo(() => {
    if (!mounted) return "";
    return new Date().toLocaleTimeString();
  }, [mounted]);

  const debateRunning = debate.active && !debate.finished;
  const showConsensus = debate.finished && ws.consensus !== null;

  // Header status colors.
  const status = useMemo(() => {
    if (ws.phase === "error") return { label: "Error", color: "#DC2626" };
    if (showConsensus) return { label: "Resolved", color: "#059669" };
    if (debateRunning) return { label: `Debating · R${debate.round || 1}`, color: "#f97316" };
    if (ws.agentA && ws.agentB) return { label: "Agents ready", color: "#3b82f6" };
    if (ws.phase === "running") return { label: "Processing", color: "#3b82f6" };
    return { label: "Awaiting", color: "#6b7280" };
  }, [ws.phase, ws.agentA, ws.agentB, debateRunning, debate.round, showConsensus]);

  // Per-agent scoreboard inputs.
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
    ? `After ${debate.round} rounds the agents reconciled their reading to ${ws.consensus.pred_class}. Calibrated head confidence: ${(ws.consensus.confidence * 100).toFixed(0)}%.`
    : "";

  return (
    <main className="flex h-screen w-screen flex-col overflow-hidden text-slate-300 font-sans" style={{ backgroundColor: "#000000" }}>
      {/* ── HEADER BAR (DICOM style) ────────────────────────────────── */}
      <header
        className="flex h-12 w-full shrink-0 items-center justify-between px-4 border-b select-none"
        style={{ backgroundColor: "#1e222b", borderColor: "#2d313c" }}
      >
        {/* Left: Patient / Specimen ID tags */}
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <img src="/logo.png" alt="Argus Vision" className="h-6 w-auto object-contain opacity-90" />
          </div>

          <div className="flex items-center gap-4 text-xs font-mono">
            <div className="flex items-center gap-2">
              <span className="text-[#a1a1a6] font-semibold">SUBJECT:</span>
              <span className="text-[#fbbf24]">ISIC_SPECIMEN_{jobId.slice(0, 8).toUpperCase()}</span>
            </div>
            <div className="h-3 w-px bg-[#2d313c]" />
            <div className="flex items-center gap-2">
              <span className="text-[#a1a1a6] font-semibold">STUDY DATE:</span>
              <span className="text-[#e5e7eb]">{formattedDate} {formattedTime}</span>
            </div>
          </div>
        </div>

        {/* Right: Actions / Status */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 rounded px-2.5 py-0.5" style={{ backgroundColor: `${status.color}15`, border: `1px solid ${status.color}40` }}>
            <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: status.color }} />
            <span className="font-mono text-[10px] font-bold uppercase tracking-wider" style={{ color: status.color }}>
              {status.label}
            </span>
          </div>

          <div className="flex items-center gap-3 text-slate-400">
            <svg viewBox="0 0 24 24" className="h-4 w-4 opacity-75" fill="none" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.02 6.02 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
            </svg>
            <svg viewBox="0 0 24 24" className="h-4 w-4 opacity-75" fill="none" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
            </svg>
          </div>
        </div>
      </header>

      {/* ── MAIN WORKSPACE ────────────────────────────────────────── */}
      <div className="flex flex-1 w-full min-h-0 overflow-hidden">
        
        {/* ── LEFT SIDEBAR: Action Toolbar + Patient Explorer ────── */}
        <aside className="flex w-[260px] shrink-0 border-r select-none" style={{ backgroundColor: "#13161c", borderColor: "#2d313c" }}>
          {/* Vertical action icons */}
          <div className="flex w-12 flex-col items-center py-4 border-r gap-5" style={{ backgroundColor: "#0d0f13", borderColor: "#2d313c" }}>
            {[
              { id: "explore", path: "M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" },
              { id: "grid", path: "M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" },
              { id: "measure", path: "M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" },
              { id: "adjust", path: "M12 3v1m0 16v1m9-9h-1M4 12H3m15.364-6.364l-.707.707M6.343 17.657l-.707.707m0-12.728l.707.707m11.314 11.314l.707.707M12 5a7 7 0 100 14 7 7 0 000-14z" },
            ].map((icon) => (
              <button key={icon.id} className="text-slate-500 hover:text-slate-200 transition-colors">
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d={icon.path} />
                </svg>
              </button>
            ))}
          </div>

          {/* Pipeline timeline explorer */}
          <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-4">
            <TimelineRail completedAt={ws.completedAt} phase={ws.phase} triggerFired={ws.triggerFired} />
          </div>
        </aside>

        {/* ── CENTER GRID: 2x2 DICOM Image Viewer ────────────────── */}
        <section className="flex-1 min-w-0 bg-[#000000] p-1 grid grid-cols-2 grid-rows-2 gap-1 border-r select-none" style={{ borderColor: "#2d313c" }}>
          
          {/* Quadrant 1: Localizer Specimen */}
          <div
            onClick={() => setSelectedViewport(1)}
            className="relative flex flex-col items-stretch overflow-hidden border cursor-pointer group"
            style={{
              borderColor: selectedViewport === 1 ? "#fbbf24" : "#1a1a1f",
              backgroundColor: "#050505"
            }}
          >
            {/* Viewport header tags */}
            <div className="absolute top-2 left-2 z-10 font-mono text-[9px] text-[#9ca3af] leading-tight pointer-events-none">
              <div>{formattedDate}</div>
              <div>STUDY: LOCALIZER</div>
              <div>3PLAN SCAN</div>
            </div>
            <div className="absolute top-2 right-2 z-10 font-mono text-[9px] text-[#fbbf24] font-bold pointer-events-none">
              SR
            </div>
            <div className="absolute top-2 left-1/2 -translate-x-1/2 z-10 font-mono text-[10px] text-[#9ca3af] tracking-wider pointer-events-none">
              CORONAL
            </div>

            {/* Main specimen image */}
            <div className="flex-1 flex items-center justify-center p-6 min-h-0">
              {sourceImage ? (
                <img
                  src={sourceImage}
                  alt="Source Specimen"
                  className="max-h-full max-w-full object-contain border"
                  style={{ borderColor: "#1f1f23" }}
                />
              ) : (
                <span className="font-mono text-[10px] text-slate-600">NO LOCALIZER TARGET</span>
              )}
            </div>

            {/* Viewport footer info */}
            <div className="absolute bottom-2 left-2 z-10 font-mono text-[9px] text-[#6b7280] leading-tight pointer-events-none">
              <div>Images: 1/1</div>
              <div>Wt: 256 / ww: 256</div>
              <div>Zoom: 100%</div>
            </div>
            <div className="absolute bottom-2 right-2 z-10 font-mono text-[9px] text-[#6b7280] leading-tight text-right pointer-events-none">
              <div>Size: 224 x 224</div>
              <div>Thick: 7.00 mm</div>
            </div>
          </div>

          {/* Quadrant 2: Agent A Attention */}
          <div
            onClick={() => setSelectedViewport(2)}
            className="relative flex flex-col items-stretch overflow-hidden border cursor-pointer group"
            style={{
              borderColor: selectedViewport === 2 ? "#fbbf24" : "#1a1a1f",
              backgroundColor: "#050505"
            }}
          >
            <div className="absolute top-2 left-2 z-10 font-mono text-[9px] text-[#9ca3af] leading-tight pointer-events-none">
              <div>AGENT A: CNN ANALYSIS</div>
              <div>SALIENCY: GRAD-CAM++</div>
              <div>LAYER: features.16</div>
            </div>
            <div className="absolute top-2 right-2 z-10 font-mono text-[9px] text-blue-500 font-bold pointer-events-none">
              AL
            </div>
            <div className="absolute top-2 left-1/2 -translate-x-1/2 z-10 font-mono text-[10px] text-[#9ca3af] tracking-wider pointer-events-none">
              SAGITTAL
            </div>

            <div className="flex-1 flex items-center justify-center p-6 min-h-0">
              {ws.attention ? (
                <HeatmapCanvas
                  b64={ws.attention.heatmap_a_b64}
                  accent={AGENT_A.color}
                  showOverlay={false}
                  alt="Agent A Heatmap"
                />
              ) : (
                <span className="font-mono text-[10px] text-slate-600">AWAITING ATTENTION MATRIX</span>
              )}
            </div>

            <div className="absolute bottom-2 left-2 z-10 font-mono text-[9px] text-[#6b7280] leading-tight pointer-events-none">
              <div>Target: {leadClass(aProbs) || "N/A"}</div>
              <div>Confidence: {(aConf * 100).toFixed(0)}%</div>
            </div>
            <div className="absolute bottom-2 right-2 z-10 font-mono text-[9px] text-[#6b7280] leading-tight text-right pointer-events-none">
              <div>Size: 224 x 224</div>
              <div>Zoom: 100%</div>
            </div>
          </div>

          {/* Quadrant 3: Agent B Attention */}
          <div
            onClick={() => setSelectedViewport(3)}
            className="relative flex flex-col items-stretch overflow-hidden border cursor-pointer group"
            style={{
              borderColor: selectedViewport === 3 ? "#fbbf24" : "#1a1a1f",
              backgroundColor: "#050505"
            }}
          >
            <div className="absolute top-2 left-2 z-10 font-mono text-[9px] text-[#9ca3af] leading-tight pointer-events-none">
              <div>AGENT B: ViT ANALYSIS</div>
              <div>SALIENCY: ATTN ROLLOUT</div>
              <div>LAYER: cls_self_attn</div>
            </div>
            <div className="absolute top-2 right-2 z-10 font-mono text-[9px] text-purple-400 font-bold pointer-events-none">
              PF
            </div>
            <div className="absolute top-2 left-1/2 -translate-x-1/2 z-10 font-mono text-[10px] text-[#9ca3af] tracking-wider pointer-events-none">
              AXIAL
            </div>

            <div className="flex-1 flex items-center justify-center p-6 min-h-0">
              {ws.attention ? (
                <HeatmapCanvas
                  b64={ws.attention.heatmap_b_b64}
                  accent={AGENT_B.color}
                  showOverlay={false}
                  alt="Agent B Heatmap"
                />
              ) : (
                <span className="font-mono text-[10px] text-slate-600">AWAITING ATTENTION MATRIX</span>
              )}
            </div>

            <div className="absolute bottom-2 left-2 z-10 font-mono text-[9px] text-[#6b7280] leading-tight pointer-events-none">
              <div>Target: {leadClass(bProbs) || "N/A"}</div>
              <div>Confidence: {(bConf * 100).toFixed(0)}%</div>
            </div>
            <div className="absolute bottom-2 right-2 z-10 font-mono text-[9px] text-[#6b7280] leading-tight text-right pointer-events-none">
              <div>Size: 224 x 224</div>
              <div>Zoom: 100%</div>
            </div>
          </div>

          {/* Quadrant 4: Disagreement / Alignment */}
          <div
            onClick={() => setSelectedViewport(4)}
            className="relative flex flex-col items-stretch overflow-hidden border cursor-pointer group"
            style={{
              borderColor: selectedViewport === 4 ? "#fbbf24" : "#1a1a1f",
              backgroundColor: "#050505"
            }}
          >
            <div className="absolute top-2 left-2 z-10 font-mono text-[9px] text-[#9ca3af] leading-tight pointer-events-none">
              <div>CROSS-ALIGNMENT DETECTOR</div>
              <div>METHOD: ANOMALY DIFF</div>
              <div>TRIGGER: JS COMPUTE</div>
            </div>
            <div className="absolute top-2 right-2 z-10 font-mono text-[9px] text-[#dc2626] font-bold pointer-events-none">
              LH
            </div>
            <div className="absolute top-2 left-1/2 -translate-x-1/2 z-10 font-mono text-[10px] text-[#9ca3af] tracking-wider pointer-events-none">
              3D RECON
            </div>

            <div className="flex-1 flex items-center justify-center p-6 min-h-0">
              {ws.attention ? (
                <HeatmapCanvas
                  b64={ws.attention.disagreement_b64}
                  bbox={ws.attention.bbox}
                  accent="#dc2626"
                  showOverlay={true}
                  alt="Disagreement Alignment"
                />
              ) : (
                <span className="font-mono text-[10px] text-slate-600">AWAITING CROSS-ALIGNMENT MATRIX</span>
              )}
            </div>

            <div className="absolute bottom-2 left-2 z-10 font-mono text-[9px] text-[#6b7280] leading-tight pointer-events-none">
              <div>Divergence: {ws.trigger ? ws.trigger.js_divergence.toFixed(4) : "0.0000"}</div>
              <div>Status: {ws.trigger?.fired ? "DEBATE TRIGGERED" : "FAST PATH"}</div>
            </div>
            <div className="absolute bottom-2 right-2 z-10 font-mono text-[9px] text-[#6b7280] leading-tight text-right pointer-events-none">
              <div>Size: 224 x 224</div>
              <div>Zoom: 100%</div>
            </div>
          </div>

        </section>

        {/* ── RIGHT PANEL:stacked "Seriler" tray (Agent, consensus & log) ── */}
        <aside className="w-[480px] shrink-0 border-l flex flex-col" style={{ backgroundColor: "#13161c", borderColor: "#2d313c" }}>
          {/* Header */}
          <div className="flex h-10 items-center justify-between px-4 border-b shrink-0" style={{ backgroundColor: "#1e222b", borderColor: "#2d313c" }}>
            <span className="font-mono text-[10px] font-bold uppercase tracking-wider text-[#a1a1a6]">
              Diagnostic Series (Seriler)
            </span>
            <span className="font-mono text-[9px] text-[#6b7280]">
              4 Series Loaded
            </span>
          </div>

          {/* Series list */}
          <div className="flex-1 overflow-y-auto p-3 flex flex-col gap-3">
            
            {/* Series 1: Consensus Verdict (The primary series) */}
            <div
              onClick={() => setActiveSeries("consensus")}
              className="rounded border p-1 cursor-pointer transition-all duration-200"
              style={{
                borderColor: activeSeries === "consensus" ? "#fbbf24" : "#2d313c",
                backgroundColor: activeSeries === "consensus" ? "#1a1f28" : "#171b22",
              }}
            >
              {/* Thumbnail header */}
              <div className="flex items-center justify-between px-2 py-1 select-none">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[10px] font-bold text-[#fbbf24]">1</span>
                  <span className="font-mono text-[10px] text-[#e5e7eb] font-semibold">SERIES: CONSENSUS_VERDICT</span>
                </div>
                <span className="font-mono text-[9px] text-[#6b7280]">ECE FITTED</span>
              </div>
              
              {/* Verdict component details inside series box */}
              {showConsensus && ws.consensus ? (
                <ConsensusVerdict
                  consensus={ws.consensus}
                  trigger={ws.trigger}
                  synthesis={recap}
                  synthesisActive={false}
                />
              ) : (
                <div className="p-4 font-mono text-[11px] text-slate-500">
                  {ws.phase === "error" ? "DIAGNOSTIC PROCESS FAILED" : "AWAITING PROCESS RESOLUTION"}
                </div>
              )}
            </div>

            {/* Series 2: Agent A analysis */}
            <div
              onClick={() => setActiveSeries("A")}
              className="rounded border p-1 cursor-pointer transition-all duration-200"
              style={{
                borderColor: activeSeries === "A" ? "#fbbf24" : "#2d313c",
                backgroundColor: activeSeries === "A" ? "#1a1f28" : "#171b22",
              }}
            >
              <div className="flex items-center justify-between px-2 py-1 select-none">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[10px] font-bold text-blue-500">2</span>
                  <span className="font-mono text-[10px] text-[#e5e7eb] font-semibold">SERIES: CNN_CLASSIFIER</span>
                </div>
                <span className="font-mono text-[9px] text-[#6b7280]">8 CLASSES</span>
              </div>
              <AgentScoreboard
                agentId="A"
                probs={aProbs}
                confidence={aConf}
                topClass={leadClass(aProbs)}
                status={statusFor("A", ws.agentA !== null)}
              />
            </div>

            {/* Series 3: Agent B analysis */}
            <div
              onClick={() => setActiveSeries("B")}
              className="rounded border p-1 cursor-pointer transition-all duration-200"
              style={{
                borderColor: activeSeries === "B" ? "#fbbf24" : "#2d313c",
                backgroundColor: activeSeries === "B" ? "#1a1f28" : "#171b22",
              }}
            >
              <div className="flex items-center justify-between px-2 py-1 select-none">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[10px] font-bold text-purple-400">3</span>
                  <span className="font-mono text-[10px] text-[#e5e7eb] font-semibold">SERIES: VIT_CLASSIFIER</span>
                </div>
                <span className="font-mono text-[9px] text-[#6b7280]">8 CLASSES</span>
              </div>
              <AgentScoreboard
                agentId="B"
                probs={bProbs}
                confidence={bConf}
                topClass={leadClass(bProbs)}
                status={statusFor("B", ws.agentB !== null)}
              />
            </div>

            {/* Series 4: Transaction Log */}
            <div
              onClick={() => setActiveSeries("log")}
              className="rounded border p-1 cursor-pointer transition-all duration-200"
              style={{
                borderColor: activeSeries === "log" ? "#fbbf24" : "#2d313c",
                backgroundColor: activeSeries === "log" ? "#1a1f28" : "#171b22",
              }}
            >
              <div className="flex items-center justify-between px-2 py-1 select-none">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[10px] font-bold text-[#6b7280]">4</span>
                  <span className="font-mono text-[10px] text-[#e5e7eb] font-semibold">SERIES: DEBATE_AUDIT_LOG</span>
                </div>
                <span className="font-mono text-[9px] text-[#6b7280]">STREAM LIVE</span>
              </div>
              {(debate.active || debate.turns.length > 0) ? (
                <DebateTranscript
                  turns={debate.turns}
                  agreement={debate.agreement}
                  round={debate.round}
                  converged={debate.converged}
                  finished={debate.finished}
                  active={debateRunning}
                  convergedClass={convergedClass}
                />
              ) : (
                <div className="p-4 font-mono text-[11px] text-slate-500">
                  DEBATE TRANSCRIPT LOG EMPTY
                </div>
              )}
            </div>

          </div>
        </aside>

      </div>
    </main>
  );
}
