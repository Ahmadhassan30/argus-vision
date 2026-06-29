/**
 * Provider-agnostic LLM streaming proxy for the live debate arguments.
 *
 * The browser never sees the API key: it POSTs an OpenAI-style chat request to
 * this route, which forwards it to a *free* OpenAI-compatible provider (Groq by
 * default, or OpenRouter) and pipes the SSE stream straight back. Both providers
 * speak the same protocol, so switching is purely env configuration.
 *
 * Environment (set in `frontend/.env.local`):
 *   LLM_PROVIDER   "groq" | "openrouter"        (default "groq")
 *   LLM_API_KEY    provider API key             (or GROQ_API_KEY / OPENROUTER_API_KEY)
 *   LLM_MODEL      model id                      (sensible per-provider default)
 *   LLM_BASE_URL   optional base URL override    (advanced)
 *
 * When no key is configured the route returns 503 with `{ error: "no_key" }`,
 * which the client treats as the signal to fall back to deterministic local
 * argument text. The UI therefore works fully offline and for free.
 */

import { NextRequest } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

interface ProviderConfig {
  url: string;
  key: string | undefined;
  model: string;
  extraHeaders: Record<string, string>;
}

/** Resolve the active provider configuration from environment variables. */
function resolveProvider(): ProviderConfig {
  const provider = (process.env.LLM_PROVIDER ?? "groq").toLowerCase();
  const override = process.env.LLM_BASE_URL;

  if (provider === "openrouter") {
    return {
      url: override ?? "https://openrouter.ai/api/v1/chat/completions",
      key: process.env.LLM_API_KEY ?? process.env.OPENROUTER_API_KEY,
      model: process.env.LLM_MODEL ?? "meta-llama/llama-3.3-70b-instruct:free",
      extraHeaders: {
        // Optional but recommended by OpenRouter for attribution.
        "HTTP-Referer": process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost",
        "X-Title": "Argus Vision",
      },
    };
  }

  // Default: Groq — fastest free tier, generous daily limits.
  return {
    url: override ?? "https://api.groq.com/openai/v1/chat/completions",
    key: process.env.LLM_API_KEY ?? process.env.GROQ_API_KEY,
    model: process.env.LLM_MODEL ?? "llama-3.3-70b-versatile",
    extraHeaders: {},
  };
}

function json(body: unknown, status: number): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

export async function POST(req: NextRequest): Promise<Response> {
  const provider = resolveProvider();

  if (!provider.key) {
    // No key configured → tell the client to use its local fallback.
    return json({ error: "no_key" }, 503);
  }

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return json({ error: "invalid_json" }, 400);
  }

  const messages = body.messages;
  if (!Array.isArray(messages) || messages.length === 0) {
    return json({ error: "missing_messages" }, 400);
  }

  const payload = {
    model: typeof body.model === "string" ? body.model : provider.model,
    messages,
    stream: true,
    temperature: typeof body.temperature === "number" ? body.temperature : 0.85,
    max_tokens: typeof body.max_tokens === "number" ? body.max_tokens : 220,
  };

  let upstream: Response;
  try {
    upstream = await fetch(provider.url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${provider.key}`,
        ...provider.extraHeaders,
      },
      body: JSON.stringify(payload),
    });
  } catch {
    return json({ error: "upstream_unreachable" }, 502);
  }

  if (!upstream.ok || !upstream.body) {
    const detail = await upstream.text().catch(() => "");
    return json({ error: "upstream_error", status: upstream.status, detail }, 502);
  }

  // Pipe the provider's SSE stream straight back to the client.
  return new Response(upstream.body, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
