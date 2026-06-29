"use client";

/**
 * ArgumentStream — the signature element. Reveals an agent's argument text
 * character-by-character at a steady cinematic cadence using requestAnimationFrame,
 * regardless of how chunkily the underlying tokens arrive. A blinking caret marks
 * live cognition. Honours reduced-motion by showing the full text at once.
 *
 * The component reveals toward whatever `text` currently is, so as new tokens grow
 * the target it keeps typing; if the text is replaced (e.g. fallback swap) it
 * clamps gracefully.
 */

import { useEffect, useRef, useState } from "react";

import { AGENTS, type AgentId } from "@/lib/constants";

interface ArgumentStreamProps {
  text: string;
  /** Drives caret colour; omit for the neutral consensus voice. */
  agentId?: AgentId;
  /** Show the caret even when fully revealed (request still in flight). */
  active?: boolean;
  /** Reveal speed in characters per second. */
  speed?: number;
  /** Render synthesis voice in italic serif. */
  serif?: boolean;
  className?: string;
}

export default function ArgumentStream({
  text,
  agentId,
  active = false,
  speed = 48,
  serif = false,
  className = "",
}: ArgumentStreamProps): React.JSX.Element {
  const [shown, setShown] = useState(0);
  const rafRef = useRef<number | null>(null);
  const lastRef = useRef<number>(0);
  const accRef = useRef<number>(0);

  const reduced =
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  useEffect(() => {
    if (reduced) {
      setShown(text.length);
      return;
    }

    // Clamp if the target shrank (text replaced).
    setShown((s) => Math.min(s, text.length));

    const tick = (now: number): void => {
      if (lastRef.current === 0) lastRef.current = now;
      const dt = (now - lastRef.current) / 1000;
      lastRef.current = now;
      accRef.current += dt * speed;

      setShown((prev) => {
        if (prev >= text.length) return prev;
        const add = Math.floor(accRef.current);
        if (add >= 1) {
          accRef.current -= add;
          return Math.min(prev + add, text.length);
        }
        return prev;
      });

      rafRef.current = requestAnimationFrame(tick);
    };

    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      lastRef.current = 0;
    };
  }, [text, speed, reduced]);

  const visible = text.slice(0, shown);
  const caretVisible = active || shown < text.length;
  const caretColor = agentId ? AGENTS[agentId].color : "var(--accent-consensus)";

  return (
    <span
      className={[
        serif ? "font-display italic" : "",
        "whitespace-pre-wrap",
        className,
      ].join(" ")}
    >
      {visible}
      {caretVisible && (
        <span
          aria-hidden
          className="ml-px inline-block animate-blink"
          style={{ color: caretColor }}
        >
          ▋
        </span>
      )}
    </span>
  );
}
