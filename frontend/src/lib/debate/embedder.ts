/**
 * Pluggable sentence embedder for retrieval-based debate text.
 *
 * Two backends, same interface:
 *   • "lexical"      — a dependency-free signed feature-hashing embedder. Works
 *                      immediately, no install, no download; good enough to rank
 *                      a curated bank by token overlap.
 *   • "transformers" — real semantic embeddings from all-MiniLM-L6-v2 (384-d) via
 *                      transformers.js, loaded from a CDN at runtime (no install)
 *                      and cached in the browser. Enable with
 *                      NEXT_PUBLIC_DEBATE_EMBEDDER=transformers. Falls back to
 *                      lexical automatically if the library or model can't load.
 *
 * The CDN import is marked `webpackIgnore` so the bundler never tries to resolve
 * it — the app builds and runs identically whether or not the model is used.
 */

export interface Embedder {
  name: string;
  dim: number;
  embed: (texts: string[]) => Promise<number[][]>;
}

const DIM = 384;

const STOP = new Set([
  "the", "a", "an", "is", "it", "to", "of", "and", "i", "you", "that", "this",
  "on", "in", "as", "at", "my", "me", "we", "be", "so", "but", "if", "for",
]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length >= 2 && !STOP.has(t));
}

/** Two cheap independent string hashes for signed feature hashing. */
function hash(str: string, seed: number): number {
  let h = seed >>> 0;
  for (let i = 0; i < str.length; i++) {
    h = Math.imul(h ^ str.charCodeAt(i), 0x01000193) >>> 0;
  }
  return h >>> 0;
}

function l2normalize(v: number[]): number[] {
  let s = 0;
  for (const x of v) s += x * x;
  const n = Math.sqrt(s) || 1;
  return v.map((x) => x / n);
}

/** Dependency-free signed feature-hashing embedder. */
export const lexicalEmbedder: Embedder = {
  name: "lexical",
  dim: DIM,
  async embed(texts: string[]): Promise<number[][]> {
    return texts.map((text) => {
      const v = new Array(DIM).fill(0);
      const toks = tokenize(text);
      for (const tok of toks) {
        const idx = hash(tok, 2166136261) % DIM;
        const sign = hash(tok, 0x9e3779b1) % 2 === 0 ? 1 : -1;
        v[idx] += sign;
        // Light bigram signal for a bit more discrimination.
      }
      return l2normalize(v);
    });
  },
};

/** Lazily-constructed transformers.js embedder (or null if unavailable). */
let transformersPromise: Promise<Embedder | null> | null = null;

async function loadTransformersEmbedder(): Promise<Embedder | null> {
  try {
    // Load transformers.js from a CDN at runtime; webpackIgnore keeps the
    // bundler from trying to resolve it, so no install/build impact.
    const url =
      process.env.NEXT_PUBLIC_TRANSFORMERS_URL ||
      "https://esm.sh/@xenova/transformers@2.17.2";
    const mod: any = await import(/* webpackIgnore: true */ url).catch(() => null);
    if (!mod || !mod.pipeline) return null;
    if (mod.env) {
      mod.env.allowLocalModels = false;
      mod.env.useBrowserCache = true;
    }
    const extractor: any = await mod.pipeline(
      "feature-extraction",
      "Xenova/all-MiniLM-L6-v2",
    );
    return {
      name: "transformers/all-MiniLM-L6-v2",
      dim: DIM,
      async embed(texts: string[]): Promise<number[][]> {
        const out: number[][] = [];
        for (const t of texts) {
          const res: any = await extractor(t, { pooling: "mean", normalize: true });
          out.push(Array.from(res.data as Float32Array));
        }
        return out;
      },
    };
  } catch {
    return null;
  }
}

let active: Embedder = lexicalEmbedder;

/**
 * Resolve the active embedder. With NEXT_PUBLIC_DEBATE_EMBEDDER=transformers it
 * attempts MiniLM and upgrades in place once loaded; otherwise (or on failure)
 * it stays lexical. Always resolves to a working embedder.
 */
export async function getEmbedder(): Promise<Embedder> {
  const want = process.env.NEXT_PUBLIC_DEBATE_EMBEDDER;
  if (want !== "transformers") return lexicalEmbedder;

  if (transformersPromise === null) {
    transformersPromise = loadTransformersEmbedder();
  }
  const t = await transformersPromise;
  if (t) active = t;
  return active;
}

/** Cosine similarity between two equal-length, ideally normalized, vectors. */
export function cosine(a: number[], b: number[]): number {
  let dot = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) dot += a[i] * b[i];
  return dot;
}
