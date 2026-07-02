"use client";

/**
 * ArgumentStream — dynamic human-like typing stream.
 *
 * Types out the argument text character-by-character with variable natural delays:
 * - standard characters: fast (12ms)
 * - spaces: brief pause (25ms)
 * - commas/semicolons: moderate pause (220ms)
 * - sentence endings (., ?, !): longer pause (500ms)
 *
 * This mimics natural human pacing during deliberation. Honours
 * prefers-reduced-motion by rendering immediately.
 */

import { useEffect, useState } from "react";
import { AGENTS, type AgentId } from "@/lib/constants";

interface ArgumentStreamProps {
  text: string;
  /** Drives caret colour; omit for neutral voice. */
  agentId?: AgentId;
  /** Whether the generation is currently active. */
  active?: boolean;
  className?: string;
}

export default function ArgumentStream({
  text,
  agentId,
  active = false,
  className = "",
}: ArgumentStreamProps): React.JSX.Element {
  const [visibleText, setVisibleText] = useState("");

  const reduced =
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  useEffect(() => {
    if (reduced) {
      setVisibleText(text);
      return;
    }

    let isMounted = true;
    let index = 0;
    let timerId: ReturnType<typeof setTimeout>;

    const typeNextChar = () => {
      if (!isMounted) return;
      if (index >= text.length) {
        setVisibleText(text);
        return;
      }

      const nextText = text.slice(0, index + 1);
      setVisibleText(nextText);
      const char = text[index];
      index++;

      // Compute natural typing delays
      let delay = 12; // Base letter delay
      if (char === " ") {
        delay = 25;
      } else if (char === "," || char === ";" || char === ":") {
        delay = 220;
      } else if (char === "." || char === "?" || char === "!") {
        // Lookahead to make sure it's sentence boundary, not decimal / acronym
        const nextChar = text[index];
        if (!nextChar || nextChar === " " || nextChar === "\n") {
          delay = 500;
        }
      }

      timerId = setTimeout(typeNextChar, delay);
    };

    typeNextChar();

    return () => {
      isMounted = false;
      clearTimeout(timerId);
    };
  }, [text, reduced]);

  const caretColor = agentId ? AGENTS[agentId].color : "var(--accent-consensus)";
  const isTyping = active && visibleText.length < text.length;

  return (
    <span className={["whitespace-pre-wrap font-mono text-[13px] leading-relaxed", className].join(" ")}>
      {visibleText}
      {isTyping && (
        <span
          aria-hidden
          className="ml-0.5 inline-block w-1.5 h-3.5 align-middle animate-pulse"
          style={{ backgroundColor: caretColor }}
        />
      )}
    </span>
  );
}
