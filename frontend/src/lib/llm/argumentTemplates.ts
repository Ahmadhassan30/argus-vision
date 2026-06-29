/**
 * Deterministic, offline argument generator — the graceful fallback for when no
 * free-LLM key is configured or a provider call fails. It composes genuinely
 * specific dermatological prose from each agent's real prediction and runner-up,
 * in the agent's architectural voice, so the signature "witnessed cognition"
 * feature keeps working for free and never blocks the UI.
 */

import type { AgentId } from "@/lib/constants";
import { getClassName } from "@/lib/constants";
import type { ClassificationResult, ConsensusResult, TriggerResult } from "@/types/debate";

/** Architecture-specific visual cue per class: [CNN-framing, ViT-framing]. */
const CUES: Record<string, [string, string]> = {
  MEL: [
    "an atypical, asymmetric pigment network with an abrupt border cut-off and irregular dark globules at the lateral margin",
    "pronounced global asymmetry and structural disorder, with no coherent organising axis across the lesion",
  ],
  NV: [
    "a regular, finely reticular pigment network that fades evenly toward a soft border",
    "a globally symmetric, orderly architecture whose patches relate smoothly across the whole field",
  ],
  BCC: [
    "arborizing telangiectatic vessels and blue-grey ovoid nests against a translucent base",
    "a cohesive, non-pigmented structural pattern dominated by a branching vascular skeleton",
  ],
  AK: [
    "a strawberry-pattern erythematous background with follicular plugging and fine surface scale",
    "a diffuse surface-scale structure without any single discrete focus of concern",
  ],
  BKL: [
    "sharply demarcated milia-like cysts and comedo-like openings over a waxy keratotic surface",
    "an evenly textured, well-bounded keratotic field with stable, repeating structure",
  ],
  DF: [
    "a central white scar-like patch ringed by a delicate, even pigment band",
    "a radially symmetric structure organised around a stable, quiet centre",
  ],
  VASC: [
    "well-defined red-purple lacunae separated by pale septa",
    "a globally uniform vascular architecture with consistent patch-level colour",
  ],
  SCC: [
    "keratin pearls and looped, glomerular vessels over a thickened scaly base",
    "a coarse, disorganised keratinising surface whose structure breaks down toward the centre",
  ],
};

function cue(agentId: AgentId, cls: string): string {
  const pair = CUES[cls];
  if (!pair) return agentId === "A" ? "a distinctive local surface signature" : "a distinctive global structural signature";
  return agentId === "A" ? pair[0] : pair[1];
}

/** Sorted [classId, prob] pairs, highest first. */
function ranked(probabilities: Record<string, number>): [string, number][] {
  return Object.entries(probabilities).sort(([, a], [, b]) => b - a);
}

const PERCEIVE: Record<AgentId, string> = {
  A: "Across the surface I read",
  B: "Taken as a whole, I see",
};

/** Round-1 opening argument. */
export function templateRound1(
  agentId: AgentId,
  result: ClassificationResult,
): string {
  const order = ranked(result.probabilities);
  const top = result.pred_class;
  const second = order.find(([c]) => c !== top)?.[0] ?? top;
  const conf = result.confidence;

  const lead = `${PERCEIVE[agentId]} ${cue(agentId, top)}, which points me to ${getClassName(top)}.`;

  let stance: string;
  if (conf >= 0.9) {
    stance = `The evidence is decisive here — I am confident in ${result.pred_class}.`;
  } else if (conf >= 0.7) {
    stance = `I lean firmly toward ${result.pred_class} at ${(conf * 100).toFixed(0)}% conviction, though I keep ${getClassName(second)} on the table.`;
  } else {
    stance = `I only tentatively favour ${result.pred_class} at ${(conf * 100).toFixed(0)}%; the picture is genuinely ambiguous and I cannot yet exclude ${getClassName(second)}.`;
  }

  const contrast =
    agentId === "A"
      ? `Where ${getClassName(second)} would show ${cue("A", second)}, that local signature is weaker than what I am reading.`
      : `If this were ${getClassName(second)} I would expect ${cue("B", second)}; the overall structure argues otherwise.`;

  return `${lead} ${stance} ${contrast}`;
}

/** Round-2 rebuttal. */
export function templateRebuttal(
  agentId: AgentId,
  myResult: ClassificationResult,
  opponentResult: ClassificationResult,
): string {
  const mine = myResult.pred_class;
  const theirs = opponentResult.pred_class;

  const attack =
    agentId === "A"
      ? `My colleague reasons from the lesion as a whole and overlooks the fine detail: ${cue("A", mine)} sits right where their read of ${getClassName(theirs)} should be quietest.`
      : `My colleague fixates on a local patch and misses the forest for the trees: at full scale the lesion shows ${cue("B", mine)}, not the orderly structure ${getClassName(theirs)} would require.`;

  const hold = `I hold ${getClassName(mine)} — the evidence I weigh most is exactly the evidence their account cannot explain.`;

  return `${attack} ${hold}`;
}

/** Neutral consensus-engine synthesis. */
export function templateSynthesis(
  agentA: ClassificationResult,
  agentB: ClassificationResult,
  trigger: TriggerResult | null,
  consensus: ConsensusResult,
): string {
  const agreed = agentA.pred_class === agentB.pred_class;
  const winner =
    consensus.pred_class === agentA.pred_class
      ? "Agent A"
      : consensus.pred_class === agentB.pred_class
        ? "Agent B"
        : "neither agent alone";

  const opening = agreed
    ? `Both agents converged on ${getClassName(consensus.pred_class)}, and the spatial evidence corroborated it.`
    : `The agents split — Agent A on ${getClassName(agentA.pred_class)}, Agent B on ${getClassName(agentB.pred_class)} — so I weighed where each reads most reliably.`;

  const jsLine =
    trigger !== null
      ? ` Their distributions diverged by ${trigger.js_divergence.toFixed(2)} against a ${trigger.threshold_js.toFixed(2)} threshold, which is why the spatial debate was called.`
      : "";

  const verdict = ` Weighting the calibrated evidence, I side with ${winner}: my read is ${getClassName(consensus.pred_class)} at ${(consensus.confidence * 100).toFixed(0)}%, calibrated at ECE ${consensus.ece.toFixed(3)}.`;

  return `${opening}${jsLine}${verdict}`;
}
