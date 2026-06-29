/**
 * Prompt construction for the live diagnostic debate.
 *
 * These build OpenAI-style chat messages (system + user) for the free LLM proxy.
 * Each agent argues in character as its architecture — Agent A (CNN) in local
 * texture / edge terms, Agent B (ViT) in global attention / structural terms —
 * grounded in its real prediction and probability distribution.
 */

import type { AgentId } from "@/lib/constants";
import { AGENTS, getClassName } from "@/lib/constants";
import type { ClassificationResult, ConsensusResult, TriggerResult } from "@/types/debate";

/** A single OpenAI-style chat message. */
export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

const ARCHITECTURE: Record<AgentId, string> = {
  A: "an EfficientNet-B4 convolutional vision system — you excel at local texture gradients, edge morphology, fine surface detail, pigment-network patterns and spatial-frequency cues",
  B: "a ViT-B/16 attention-based vision system — you excel at global context, long-range spatial relationships, holistic structural symmetry and patch-level semantic coherence",
};

const EMPHASIS: Record<AgentId, string> = {
  A: "Emphasise: texture gradients, local feature responses, edge and border morphology, pigment-network irregularity, colour-channel detail.",
  B: "Emphasise: global attention patterns, patch-to-patch relationships, contextual coherence, structural (a)symmetry at scale, overall architectural impression.",
};

/** Build the in-character system prompt for an agent. */
export function systemPrompt(agentId: AgentId): string {
  return [
    `You are Agent ${agentId}, ${ARCHITECTURE[agentId]}.`,
    "",
    "You are presenting at a live diagnostic debate over a single dermoscopic skin-lesion image, like a clinician at grand rounds. Speak in the first person, in character, as though you are directly perceiving the lesion right now.",
    EMPHASIS[agentId],
    "",
    "Rules:",
    "- 3 to 4 sentences. Tight, specific, confident but honest.",
    "- Name concrete visual evidence that drives your read, and the one alternative you weighed.",
    "- If your confidence is below 90%, say plainly where the ambiguity lives.",
    "- Never say 'neural network', 'model', 'I was trained', 'dataset', or 'probability vector'. You see; you do not compute.",
    "- Use real dermatological vocabulary (asymmetry, atypical network, blue-white veil, milia-like cysts, arborizing vessels, regression structures) where it fits the evidence.",
  ].join("\n");
}

/** Format a probability distribution as a sorted, readable block. */
function distributionBlock(probabilities: Record<string, number>): string {
  return Object.entries(probabilities)
    .sort(([, a], [, b]) => b - a)
    .map(
      ([cls, p]) =>
        `  ${cls} (${getClassName(cls)}): ${(p * 100).toFixed(1)}%`,
    )
    .join("\n");
}

/** Round-1 opening argument for one agent. */
export function round1Messages(
  agentId: AgentId,
  result: ClassificationResult,
): ChatMessage[] {
  const user = [
    "I am examining the dermoscopic lesion.",
    "",
    `My leading read: ${result.pred_class} — ${getClassName(result.pred_class)} (${(result.confidence * 100).toFixed(1)}% confidence).`,
    "",
    "How the evidence distributes across the differential:",
    distributionBlock(result.probabilities),
    "",
    "State your diagnostic argument for this read: the specific visual evidence behind it, and the alternative you considered and set aside.",
  ].join("\n");

  return [
    { role: "system", content: systemPrompt(agentId) },
    { role: "user", content: user },
  ];
}

/** Round-2 rebuttal: attack the opponent's weakest point, hold your read. */
export function rebuttalMessages(
  agentId: AgentId,
  myResult: ClassificationResult,
  opponentResult: ClassificationResult,
): ChatMessage[] {
  const opponentId: AgentId = agentId === "A" ? "B" : "A";
  const user = [
    `Your opponent, Agent ${opponentId} (${AGENTS[opponentId].name}), argues for ${opponentResult.pred_class} — ${getClassName(opponentResult.pred_class)} at ${(opponentResult.confidence * 100).toFixed(1)}% confidence.`,
    "",
    `You hold your read of ${myResult.pred_class} — ${getClassName(myResult.pred_class)} at ${(myResult.confidence * 100).toFixed(1)}%.`,
    "",
    "Write a 2 to 3 sentence rebuttal. Attack the single weakest point in their reasoning from your architectural vantage point, and say what they are missing that you can see. Do not concede; maintain your read.",
  ].join("\n");

  return [
    { role: "system", content: systemPrompt(agentId) },
    { role: "user", content: user },
  ];
}

/** Neutral consensus-engine synthesis after fusion. */
export function synthesisMessages(
  agentA: ClassificationResult,
  agentB: ClassificationResult,
  trigger: TriggerResult | null,
  consensus: ConsensusResult,
): ChatMessage[] {
  const system = [
    "You are the Argus Vision Consensus Engine — a calibrated fusion system that has just resolved a diagnostic debate between two AI agents.",
    "Speak in a measured, neutral, authoritative voice. You weigh evidence; you do not posture.",
    "Two to three sentences. Be precise about which evidence you found most reliable and why your final read differs from, or aligns with, the agents. Do not hedge excessively — you have decided.",
    "Never mention 'model', 'neural network', or 'probability vector'.",
  ].join("\n");

  const jsLine =
    trigger !== null
      ? `Jensen–Shannon divergence between the agents was ${trigger.js_divergence.toFixed(2)} (threshold ${trigger.threshold_js.toFixed(2)}).`
      : "The agents were close enough that the fast path was taken.";

  const user = [
    `Agent A (EfficientNet-B4) argued for ${agentA.pred_class} — ${getClassName(agentA.pred_class)} at ${(agentA.confidence * 100).toFixed(1)}%.`,
    `Agent B (ViT-B/16) argued for ${agentB.pred_class} — ${getClassName(agentB.pred_class)} at ${(agentB.confidence * 100).toFixed(1)}%.`,
    jsLine,
    `Your calibrated consensus: ${consensus.pred_class} — ${getClassName(consensus.pred_class)} at ${(consensus.confidence * 100).toFixed(1)}% (temperature ${consensus.temperature.toFixed(2)}, ECE ${consensus.ece.toFixed(3)}).`,
    "",
    "Explain what you weighed to reach this conclusion.",
  ].join("\n");

  return [
    { role: "system", content: system },
    { role: "user", content: user },
  ];
}
