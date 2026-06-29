/**
 * Client-side argument streamer.
 *
 * POSTs an OpenAI-style chat request to the `/api/debate-llm` proxy and parses
 * the streamed SSE deltas, reporting the *accumulated* text after each token so
 * the ArgumentStream component can reveal it at a steady cinematic cadence.
 *
 * Resilience is the whole point: if the proxy reports no key (503), the provider
 * errors, the network drops, or the stream yields nothing, it falls back to the
 * deterministic local `fallback` text — so the feature works for free and the UI
 * never stalls. Aborting (e.g. unmount) preserves whatever streamed so far.
 */

import type { ChatMessage } from "@/lib/llm/prompts";

export interface StreamArgumentOptions {
  messages: ChatMessage[];
  /** Deterministic text to emit if the live provider is unavailable. */
  fallback: string;
  /** Called with the full accumulated text after each update. */
  onText: (fullText: string) => void;
  /** Optional abort signal (cancels the request on unmount). */
  signal?: AbortSignal;
  /** Optional model override forwarded to the proxy. */
  model?: string;
}

/** Result of a stream: the final text and whether it came from the LLM. */
export interface StreamArgumentResult {
  text: string;
  source: "llm" | "fallback";
}

function extractDelta(json: unknown): string {
  const choices = (json as { choices?: Array<{ delta?: { content?: string } }> })
    .choices;
  return choices?.[0]?.delta?.content ?? "";
}

export async function streamArgument(
  opts: StreamArgumentOptions,
): Promise<StreamArgumentResult> {
  const { messages, fallback, onText, signal, model } = opts;

  const useFallback = (): StreamArgumentResult => {
    onText(fallback);
    return { text: fallback, source: "fallback" };
  };

  let res: Response;
  try {
    res = await fetch("/api/debate-llm", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages, model }),
      signal,
    });
  } catch {
    if (signal?.aborted) return { text: "", source: "fallback" };
    return useFallback();
  }

  if (!res.ok || !res.body) {
    return useFallback();
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let acc = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const raw of lines) {
        const line = raw.trim();
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") {
          return acc ? { text: acc, source: "llm" } : useFallback();
        }
        try {
          const delta = extractDelta(JSON.parse(data));
          if (delta) {
            acc += delta;
            onText(acc);
          }
        } catch {
          // Ignore non-JSON keepalive/comment frames.
        }
      }
    }
  } catch {
    if (signal?.aborted) return { text: acc, source: acc ? "llm" : "fallback" };
    // Mid-stream failure: keep partial text if we have it, else fall back.
    return acc ? { text: acc, source: "llm" } : useFallback();
  }

  return acc ? { text: acc, source: "llm" } : useFallback();
}
