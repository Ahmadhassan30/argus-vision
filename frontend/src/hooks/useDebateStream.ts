"use client";

/**
 * useDebateStream — subscribe to the Argus Vision debate WebSocket for one job
 * and project the stream into a flat {@link DebateState} via {@link debateReducer}.
 *
 * Beyond the original hook this version:
 *   • hydrates from the full {@link JobResult} snapshot the backend replays on
 *     connect (so reconnecting mid-pipeline catches up instead of starting blank);
 *   • drives a scripted run when `NEXT_PUBLIC_DEBATE_MOCK_MODE=true`, exercising
 *     the exact same reducer path as a live socket;
 *   • stamps each milestone with its completion time for the timeline rail.
 *
 * It owns the socket lifecycle, reconnects with bounded exponential backoff, and
 * stops reconnecting once the job reaches a terminal status.
 */

import { useCallback, useEffect, useReducer, useRef } from "react";

import type { DebateEvent } from "@/types/debate";
import {
  createInitialState,
  debateReducer,
  isJobSnapshot,
  isTerminalStatus,
  type DebateState,
} from "@/lib/debateReducer";
import { MOCK_MODE, buildMockSequence } from "@/lib/mockDebate";

/** Base WebSocket URL; `${WS_URL}/debate/${jobId}` is the per-job endpoint. */
const WS_URL: string = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost/ws";

/** Maximum number of automatic reconnection attempts after a non-clean close. */
const MAX_RETRIES = 3;

/** Backoff schedule (milliseconds) indexed by zero-based retry attempt. */
const BACKOFF_MS: readonly number[] = [500, 1000, 2000];

/** The public hook return type: the state plus a manual disconnect handle. */
export type UseDebateStreamReturn = DebateState & {
  disconnect: () => void;
};

/** Narrow a parsed payload to the runtime-only `ping` keep-alive frame. */
function isPing(value: unknown): boolean {
  return (
    typeof value === "object" &&
    value !== null &&
    (value as { type?: unknown }).type === "ping"
  );
}

/**
 * Subscribe to the debate stream for `jobId`.
 *
 * @param jobId - the classification job identifier to stream.
 */
export function useDebateStream(jobId: string): UseDebateStreamReturn {
  const [state, dispatch] = useReducer(debateReducer, undefined, createInitialState);

  const socketRef = useRef<WebSocket | null>(null);
  const retryCountRef = useRef(0);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const manualCloseRef = useRef(false);
  const statusRef = useRef(state.status);

  statusRef.current = state.status;

  const clearRetryTimer = useCallback((): void => {
    if (retryTimerRef.current !== null) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
  }, []);

  const disconnect = useCallback((): void => {
    manualCloseRef.current = true;
    clearRetryTimer();
    const socket = socketRef.current;
    if (socket !== null) {
      socket.onopen = null;
      socket.onmessage = null;
      socket.onerror = null;
      socket.onclose = null;
      if (
        socket.readyState === WebSocket.OPEN ||
        socket.readyState === WebSocket.CONNECTING
      ) {
        socket.close();
      }
      socketRef.current = null;
    }
    dispatch({ kind: "disconnected" });
  }, [clearRetryTimer]);

  // --- Mock mode: scripted run, no socket. --------------------------------
  useEffect(() => {
    if (!jobId || !MOCK_MODE) return undefined;

    dispatch({ kind: "reset" });
    dispatch({ kind: "connected", at: Date.now() });

    const timers: ReturnType<typeof setTimeout>[] = [];
    for (const step of buildMockSequence()) {
      timers.push(
        setTimeout(() => {
          dispatch({ kind: "event", event: step.payload as DebateEvent, at: Date.now() });
        }, step.delayMs),
      );
    }
    return () => {
      for (const t of timers) clearTimeout(t);
    };
  }, [jobId]);

  // --- Live mode: WebSocket with snapshot hydration + reconnect. ----------
  useEffect(() => {
    if (!jobId || MOCK_MODE) return undefined;

    manualCloseRef.current = false;
    retryCountRef.current = 0;
    statusRef.current = "queued";
    dispatch({ kind: "reset" });

    const connect = (): void => {
      if (manualCloseRef.current) return;

      const socket = new WebSocket(`${WS_URL}/debate/${jobId}`);
      socketRef.current = socket;

      socket.onopen = (): void => {
        retryCountRef.current = 0;
        dispatch({ kind: "connected", at: Date.now() });
      };

      socket.onmessage = (message: MessageEvent<string>): void => {
        let parsed: unknown;
        try {
          parsed = JSON.parse(message.data);
        } catch {
          return;
        }
        if (isPing(parsed)) return;

        const at = Date.now();
        if (isJobSnapshot(parsed)) {
          dispatch({ kind: "snapshot", job: parsed, at });
        } else {
          dispatch({ kind: "event", event: parsed as DebateEvent, at });
        }
      };

      socket.onerror = (): void => {
        // Surfaced via the subsequent onclose; no state change here.
      };

      socket.onclose = (): void => {
        socketRef.current = null;
        dispatch({ kind: "disconnected" });

        if (manualCloseRef.current || isTerminalStatus(statusRef.current)) return;

        if (retryCountRef.current < MAX_RETRIES) {
          const delay =
            BACKOFF_MS[retryCountRef.current] ?? BACKOFF_MS[BACKOFF_MS.length - 1];
          retryCountRef.current += 1;
          clearRetryTimer();
          retryTimerRef.current = setTimeout(connect, delay);
        }
      };
    };

    connect();

    return (): void => {
      manualCloseRef.current = true;
      clearRetryTimer();
      const socket = socketRef.current;
      if (socket !== null) {
        socket.onopen = null;
        socket.onmessage = null;
        socket.onerror = null;
        socket.onclose = null;
        if (
          socket.readyState === WebSocket.OPEN ||
          socket.readyState === WebSocket.CONNECTING
        ) {
          socket.close();
        }
        socketRef.current = null;
      }
    };
  }, [jobId, clearRetryTimer]);

  return { ...state, disconnect };
}
