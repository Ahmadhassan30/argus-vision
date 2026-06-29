/**
 * Belief-revision dynamics for the two-agent debate.
 *
 * The "debate" is, at heart, two probability distributions over the 8 ISIC
 * classes being argued toward agreement. Each turn the *speaking* agent revises
 * its belief via a confidence-weighted **logarithmic opinion pool** relaxed by a
 * learning rate (Gauss–Seidel style — only the speaker moves, so it reads as
 * genuine turn-taking). The shared target also absorbs the spatial evidence and,
 * once available, the calibrated consensus, so the debate provably contracts
 * (Jensen–Shannon divergence shrinks each round) and lands on the real result.
 *
 * References in spirit: DeGroot consensus + logarithmic opinion pooling; the
 * less-confident agent yields more, which is what makes one "concede" naturally.
 */

import { CLASS_ORDER } from "@/lib/constants";

const N = CLASS_ORDER.length;
const EPS = 1e-6;

/** A belief = probability vector in canonical CLASS_ORDER order. */
export type Vec = number[];

/** Convert a class→prob map to a normalized vector in canonical order. */
export function toVec(probs: Record<string, number>): Vec {
  const v = CLASS_ORDER.map((c) => Math.max(0, probs[c] ?? 0));
  return normalize(v);
}

/** Convert a vector back to a class→prob map. */
export function toRecord(v: Vec): Record<string, number> {
  const out: Record<string, number> = {};
  CLASS_ORDER.forEach((c, i) => {
    out[c] = v[i];
  });
  return out;
}

/** L1-normalize a non-negative vector to a probability distribution. */
export function normalize(v: Vec): Vec {
  let s = 0;
  for (const x of v) s += x;
  if (s <= 0) return new Array(v.length).fill(1 / v.length);
  return v.map((x) => x / s);
}

/** Softmax of a log-space vector. */
function softmax(logv: Vec): Vec {
  const m = Math.max(...logv);
  const ex = logv.map((x) => Math.exp(x - m));
  return normalize(ex);
}

/** Top class id and its probability for a belief vector. */
export function top(v: Vec): { cls: string; p: number; index: number } {
  let bi = 0;
  for (let i = 1; i < v.length; i++) if (v[i] > v[bi]) bi = i;
  return { cls: CLASS_ORDER[bi], p: v[bi], index: bi };
}

/** Shannon entropy in bits. */
export function entropyBits(v: Vec): number {
  let h = 0;
  for (const p of v) if (p > 0) h -= p * Math.log2(p);
  return h;
}

/** Jensen–Shannon divergence (bits, base-2 → range 0..1) between two beliefs. */
export function jsDivergence(p: Vec, q: Vec): number {
  const m = p.map((x, i) => 0.5 * (x + q[i]));
  const kl = (a: Vec, b: Vec): number => {
    let s = 0;
    for (let i = 0; i < a.length; i++) {
      if (a[i] > 0) s += a[i] * Math.log2((a[i] + EPS) / (b[i] + EPS));
    }
    return s;
  };
  return Math.max(0, 0.5 * kl(p, m) + 0.5 * kl(q, m));
}

/** Tuning constants for the revision dynamics. */
export interface ReviseParams {
  /** Base relaxation rate (max fraction the speaker moves per turn). */
  alphaBase: number;
  /** Weight of the spatial-evidence log-bias in the shared target. */
  lambdaEvidence: number;
  /** Weight of the consensus attractor (when known) in the shared target. */
  lambdaConsensus: number;
}

export const DEFAULT_PARAMS: ReviseParams = {
  alphaBase: 0.6,
  lambdaEvidence: 0.5,
  lambdaConsensus: 1.0,
};

/** What kind of rhetorical move a revision represents. */
export type Move = "open" | "press" | "rebut" | "soften" | "concede" | "agree";

/** Result of one belief revision. */
export interface Revision {
  /** The speaker's new belief. */
  next: Vec;
  /** JS divergence to the opponent after the move. */
  js: number;
  /** Classified rhetorical move. */
  move: Move;
  /** The speaker's top class after revising. */
  topClass: string;
  /** The speaker's confidence (max prob) after revising. */
  confidence: number;
  /** Whether the speaker's leading class flipped to the opponent's. */
  flipped: boolean;
  /** How far (L1) the belief shifted this turn. */
  shift: number;
}

interface ReviseInput {
  /** The speaker's current belief. */
  self: Vec;
  /** The opponent's current belief. */
  opp: Vec;
  /** Optional spatial-evidence log-bias vector (favoring a class). */
  evidence?: Vec | null;
  /** Optional consensus attractor distribution (the real result). */
  consensus?: Vec | null;
  /** Is this the speaker's very first statement (→ "open")? */
  opening?: boolean;
  /** Convergence threshold on JS divergence (to label "agree"). */
  convergeAt?: number;
  params?: ReviseParams;
}

/**
 * Revise the speaker's belief toward a confidence-weighted shared target.
 *
 * The target T = softmax( (cS·log self + cO·log opp)/(cS+cO)
 *                          + λe·evidence + λc·log consensus ), and the speaker
 * relaxes a fraction α toward T, with α larger when the *opponent* is the more
 * confident one — so the weaker-evidence agent yields, as a person would.
 */
export function revise(input: ReviseInput): Revision {
  const params = input.params ?? DEFAULT_PARAMS;
  const { self, opp } = input;
  const convergeAt = input.convergeAt ?? 0.03;

  const cS = top(self).p;
  const cO = top(opp).p;
  const wSum = cS + cO + EPS;

  const logSelf = self.map((x) => Math.log(x + EPS));
  const logOpp = opp.map((x) => Math.log(x + EPS));

  // Confidence-weighted log-pool of the two beliefs.
  const logTarget = new Array(N).fill(0).map((_, i) => (cS * logSelf[i] + cO * logOpp[i]) / wSum);

  // Spatial evidence nudges the target toward the supported class.
  if (input.evidence) {
    for (let i = 0; i < N; i++) logTarget[i] += params.lambdaEvidence * input.evidence[i];
  }
  // Consensus, once known, pulls the target toward the real result.
  if (input.consensus) {
    for (let i = 0; i < N; i++) {
      logTarget[i] += params.lambdaConsensus * Math.log(input.consensus[i] + EPS);
    }
  }
  const target = softmax(logTarget);

  // The less-confident speaker moves more.
  const alpha = params.alphaBase * (cO / wSum);

  const logNext = self.map((x, i) => (1 - alpha) * Math.log(x + EPS) + alpha * Math.log(target[i] + EPS));
  const next = softmax(logNext);

  const js = jsDivergence(next, opp);
  const beforeTop = top(self);
  const afterTop = top(next);
  const flipped = afterTop.cls !== beforeTop.cls && afterTop.cls === top(opp).cls;
  let shift = 0;
  for (let i = 0; i < N; i++) shift += Math.abs(next[i] - self[i]);

  const move = classifyMove({
    opening: input.opening ?? false,
    js,
    convergeAt,
    flipped,
    shift,
    confDelta: afterTop.p - beforeTop.p,
  });

  return { next, js, move, topClass: afterTop.cls, confidence: afterTop.p, flipped, shift };
}

function classifyMove(a: {
  opening: boolean;
  js: number;
  convergeAt: number;
  flipped: boolean;
  shift: number;
  confDelta: number;
}): Move {
  if (a.opening) return "open";
  if (a.js <= a.convergeAt) return "agree";
  if (a.flipped) return "concede";
  if (a.shift >= 0.18) return "soften";
  if (a.confDelta > 0.01) return "press";
  return "rebut";
}

/**
 * Build a spatial-evidence log-bias vector from the attention region stats.
 *
 * A higher, more concentrated region mean for an agent means the contested
 * region supports *that agent's* class. We translate the gap into a gentle log
 * bias toward each agent's leading class.
 */
export function evidenceBias(
  statsA: Record<string, number> | undefined,
  statsB: Record<string, number> | undefined,
  topA: string,
  topB: string,
): Vec | null {
  if (!statsA || !statsB) return null;
  const mA = statsA.mean ?? 0;
  const mB = statsB.mean ?? 0;
  const gap = mA - mB; // >0 → favors A's class, <0 → favors B's class
  const bias = new Array(N).fill(0);
  const ia = CLASS_ORDER.indexOf(topA);
  const ib = CLASS_ORDER.indexOf(topB);
  if (ia >= 0) bias[ia] += Math.max(0, gap);
  if (ib >= 0) bias[ib] += Math.max(0, -gap);
  return bias;
}
