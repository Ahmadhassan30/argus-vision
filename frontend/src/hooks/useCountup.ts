"use client";

/**
 * useCountup — animate a number from 0 (or a previous value) to `target` with
 * requestAnimationFrame and an easing curve. Used for confidence percentages,
 * gauge values, and probability readouts. Honours `prefers-reduced-motion` by
 * snapping straight to the target.
 */

import { useEffect, useRef, useState } from "react";

/** Expo-out easing — fast start, gentle settle. */
export function easeOutExpo(t: number): number {
  return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
}

/** Back-out easing — slight overshoot, for gauges that "snap" into place. */
export function easeOutBack(t: number): number {
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
}

interface CountupOptions {
  /** Duration in ms (default 1000). */
  duration?: number;
  /** Easing function mapping 0..1 → 0..1 (default expo-out). */
  easing?: (t: number) => number;
  /** Delay before starting, in ms (default 0). */
  delay?: number;
}

/**
 * Returns a value that animates toward `target` whenever `target` changes.
 *
 * @param target - the destination value.
 * @param options - duration, easing and delay.
 */
export function useCountup(target: number, options: CountupOptions = {}): number {
  const { duration = 1000, easing = easeOutExpo, delay = 0 } = options;
  const [value, setValue] = useState(0);
  const fromRef = useRef(0);
  const rafRef = useRef<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const reduced =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (reduced) {
      setValue(target);
      fromRef.current = target;
      return;
    }

    const from = fromRef.current;
    const start = performance.now() + delay;

    const tick = (now: number): void => {
      const elapsed = now - start;
      if (elapsed < 0) {
        rafRef.current = requestAnimationFrame(tick);
        return;
      }
      const t = Math.min(elapsed / duration, 1);
      const next = from + (target - from) * easing(t);
      setValue(next);
      if (t < 1) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        fromRef.current = target;
      }
    };

    timerRef.current = setTimeout(() => {
      rafRef.current = requestAnimationFrame(tick);
    }, 0);

    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      if (timerRef.current !== null) clearTimeout(timerRef.current);
      fromRef.current = target;
    };
  }, [target, duration, easing, delay]);

  return value;
}
