/**
 * Scripted debate for local development without a backend.
 *
 * Enabled with `NEXT_PUBLIC_DEBATE_MOCK_MODE=true`. The hook feeds each payload
 * through the exact same code path as a real WebSocket frame, so the mock
 * exercises the genuine reducer and rendering. Heatmaps are synthesised in the
 * browser (see {@link makeMockHeatmap}) so the spatial-attention panel looks
 * real offline. Timings are tuned so the run, plus streamed arguments, feels
 * like watching genuine deliberation rather than a canned animation.
 */

import type { AgentResult, AttentionResult } from "@/types/debate";
import { makeMockHeatmap, makeMockSource } from "@/lib/mockHeatmap";

/** Whether mock mode is enabled via the public env flag. */
export const MOCK_MODE: boolean =
  process.env.NEXT_PUBLIC_DEBATE_MOCK_MODE === "true";

/** One scripted frame: an opaque payload dispatched after `delayMs` from start. */
export interface MockStep {
  delayMs: number;
  /** A DebateEvent or JobResult snapshot, handled like a parsed WS message. */
  payload: unknown;
}

const PROBS_A: Record<string, number> = {
  MEL: 0.72,
  NV: 0.1,
  BCC: 0.06,
  AK: 0.04,
  BKL: 0.04,
  DF: 0.02,
  VASC: 0.01,
  SCC: 0.01,
};

const PROBS_B: Record<string, number> = {
  MEL: 0.05,
  NV: 0.12,
  BCC: 0.08,
  AK: 0.07,
  BKL: 0.61,
  DF: 0.04,
  VASC: 0.02,
  SCC: 0.01,
};

const PROBS_CONSENSUS: Record<string, number> = {
  MEL: 0.79,
  NV: 0.08,
  BCC: 0.04,
  AK: 0.03,
  BKL: 0.03,
  DF: 0.02,
  VASC: 0.01,
  SCC: 0.0,
};

function agentResult(
  id: string,
  predClass: string,
  confidence: number,
  probabilities: Record<string, number>,
  heatmap: string | null,
): AgentResult {
  return {
    agent_id: id,
    result: { pred_class: predClass, confidence, probabilities },
    heatmap_b64: heatmap,
  };
}

/**
 * Build the scripted sequence. Called in the browser so heatmaps can be
 * synthesised; falls back to empty strings (→ styled placeholders) on the server.
 */
export function buildMockSequence(): MockStep[] {
  // Agent A (CNN) fixates on the lateral margin; Agent B (ViT) on the centre —
  // their disagreement is exactly that spatial mismatch.
  const heatmapA = makeMockHeatmap({
    blobs: [
      { cx: 0.34, cy: 0.46, r: 0.16, peak: 1.0 },
      { cx: 0.28, cy: 0.62, r: 0.1, peak: 0.6 },
    ],
  });
  const heatmapB = makeMockHeatmap({
    blobs: [
      { cx: 0.52, cy: 0.5, r: 0.22, peak: 0.9 },
      { cx: 0.62, cy: 0.4, r: 0.12, peak: 0.55 },
    ],
  });
  const disagreement = makeMockHeatmap({
    base: false,
    blobs: [
      { cx: 0.34, cy: 0.46, r: 0.14, peak: 0.95 },
      { cx: 0.55, cy: 0.5, r: 0.16, peak: 0.8 },
    ],
  });
  const source = makeMockSource();

  const attention: AttentionResult = {
    heatmap_a_b64: heatmapA,
    heatmap_b_b64: heatmapB,
    disagreement_b64: disagreement,
    bbox: { x1: 66, y1: 96, x2: 168, y2: 168 },
    region_stats_a: { mean: 0.58, std: 0.12, max: 0.94 },
    region_stats_b: { mean: 0.41, std: 0.19, max: 0.88 },
  };

  return [
    { delayMs: 700, payload: { type: "agents_running" } },
    {
      delayMs: 4200,
      payload: {
        type: "agents_done",
        agent_a: agentResult("agent_a", "MEL", 0.72, PROBS_A, null),
        agent_b: agentResult("agent_b", "BKL", 0.61, PROBS_B, null),
      },
    },
    {
      delayMs: 8200,
      payload: {
        type: "trigger_evaluated",
        result: {
          fired: true,
          js_divergence: 0.38,
          entropy_a: 0.95,
          entropy_b: 1.12,
          threshold_js: 0.25,
          threshold_entropy: 0.8,
        },
      },
    },
    {
      // Second agents_done carries the freshly computed heatmaps.
      delayMs: 12600,
      payload: {
        type: "agents_done",
        agent_a: agentResult("agent_a", "MEL", 0.72, PROBS_A, source),
        agent_b: agentResult("agent_b", "BKL", 0.61, PROBS_B, heatmapB),
      },
    },
    {
      delayMs: 13200,
      payload: { type: "attention_computed", result: attention },
    },
    {
      delayMs: 18500,
      payload: {
        type: "consensus_done",
        result: {
          pred_class: "MEL",
          confidence: 0.79,
          probabilities: PROBS_CONSENSUS,
          temperature: 1.15,
          ece: 0.042,
        },
      },
    },
  ];
}
