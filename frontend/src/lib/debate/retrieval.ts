/**
 * Retrieval-based argument generation (no LLM).
 *
 * Given the live turn context — who is speaking, the rhetorical move the belief
 * revision implies, and each agent's current leading class — this builds a query,
 * embeds it, and ranks the curated argument bank by semantic similarity, skipping
 * recently-used lines so the conversation never repeats. The chosen template's
 * placeholders are then filled with the actual classes and the opponent's read,
 * yielding a coherent, on-topic, opponent-referencing turn.
 */

import { getClassName } from "@/lib/constants";
import type { Move } from "@/lib/debate/beliefs";
import {
  ARGUMENT_BANK,
  CLASS_CUE,
  OPP_ARCH,
  type ArgFragment,
} from "@/lib/debate/argumentBank";
import { cosine, getEmbedder, type Embedder } from "@/lib/debate/embedder";

export interface TurnContext {
  agent: "A" | "B";
  move: Move;
  /** Speaker's current leading class code. */
  selfClass: string;
  /** Opponent's current leading class code. */
  oppClass: string;
  /** Ids used recently, to avoid repetition. */
  recentIds: string[];
}

/** Remove {placeholders} for embedding (the literal words carry the signal). */
function strip(text: string): string {
  return text.replace(/\{[^}]+\}/g, " ");
}

/** Fill a fragment's placeholders from the turn context. */
export function fillTemplate(text: string, ctx: TurnContext): string {
  const selfName = getClassName(ctx.selfClass);
  const oppName = getClassName(ctx.oppClass);
  const map: Record<string, string> = {
    self: selfName,
    selfCode: ctx.selfClass,
    opp: oppName,
    oppCode: ctx.oppClass,
    selfCue: CLASS_CUE[ctx.selfClass] ?? "its hallmark pattern",
    oppCue: CLASS_CUE[ctx.oppClass] ?? "that pattern",
    oppArch: OPP_ARCH[ctx.agent],
  };
  return text.replace(/\{([^}]+)\}/g, (_, key: string) => map[key] ?? "");
}

// Per-embedder cache of fragment vectors.
let cacheKey = "";
let fragmentVectors: number[][] = [];

async function ensureFragmentVectors(embedder: Embedder): Promise<number[][]> {
  if (cacheKey === embedder.name && fragmentVectors.length === ARGUMENT_BANK.length) {
    return fragmentVectors;
  }
  const docs = ARGUMENT_BANK.map((f) => `${strip(f.text)} ${f.move} ${f.evidence}`);
  fragmentVectors = await embedder.embed(docs);
  cacheKey = embedder.name;
  return fragmentVectors;
}

/** Candidate fragments for a move: prefer the speaker's voice, fall back gracefully. */
function candidates(ctx: TurnContext): ArgFragment[] {
  const byMoveAndAgent = ARGUMENT_BANK.filter(
    (f) => f.move === ctx.move && (f.agent === ctx.agent || f.agent === "any"),
  );
  if (byMoveAndAgent.length > 0) return byMoveAndAgent;
  const byMove = ARGUMENT_BANK.filter((f) => f.move === ctx.move);
  if (byMove.length > 0) return byMove;
  return ARGUMENT_BANK.filter((f) => f.agent === ctx.agent || f.agent === "any");
}

export interface RetrievedTurn {
  id: string;
  text: string;
}

/**
 * Retrieve and fill the best argument line for this turn.
 *
 * Resilient: any failure (e.g. embedder error) falls back to a deterministic
 * choice so a turn is always produced.
 */
export async function retrieveTurn(ctx: TurnContext): Promise<RetrievedTurn> {
  const pool = candidates(ctx);

  try {
    const embedder = await getEmbedder();
    const vectors = await ensureFragmentVectors(embedder);

    const query = [
      ctx.move,
      getClassName(ctx.selfClass),
      CLASS_CUE[ctx.selfClass] ?? "",
      "versus",
      getClassName(ctx.oppClass),
      CLASS_CUE[ctx.oppClass] ?? "",
    ].join(" ");
    const [qVec] = await embedder.embed([query]);

    const recent = new Set(ctx.recentIds.slice(-6));
    const ranked = pool
      .map((f) => {
        const idx = ARGUMENT_BANK.indexOf(f);
        return { f, score: cosine(qVec, vectors[idx]) };
      })
      .sort((a, b) => b.score - a.score);

    const fresh = ranked.find((r) => !recent.has(r.f.id)) ?? ranked[0];
    return { id: fresh.f.id, text: fillTemplate(fresh.f.text, ctx) };
  } catch {
    // Deterministic fallback: first non-recent candidate.
    const recent = new Set(ctx.recentIds.slice(-6));
    const pick = pool.find((f) => !recent.has(f.id)) ?? pool[0];
    return { id: pick.id, text: fillTemplate(pick.text, ctx) };
  }
}
