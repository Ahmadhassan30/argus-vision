"use client";

/**
 * useDebateStream — a React hook that subscribes to the Argus Vision debate
 * WebSocket for a single classification job and projects the stream of
 * {@link DebateEvent} payloads into a flat, immutable {@link DebateState}.
 *
 * The hook owns the socket lifecycle: it opens the connection inside an effect
 * keyed on `jobId`, relays incoming events into component state, and tears the
 * socket down on cleanup or via the returned `disconnect` callback. It also
 * implements bounded exponential-backoff reconnection (up to three attempts)
 * that is suppressed once the job has reached a terminal status.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import type {
  AgentResult,
  AttentionResult,
  ConsensusResult,
  DebateEvent,
  JobStatus,
  TriggerResult,
} from "@/types/debate";

/** Base WebSocket URL; `${WS_URL}/debate/${jobId}` is the per-job endpoint. */
const WS_URL: string = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost/ws";

/** Maximum number of automatic reconnection attempts after a non-clean close. */
const MAX_RETRIES = 3;

/** Backoff schedule (milliseconds) indexed by zero-based retry attempt. */
const BACKOFF_MS: readonly number[] = [500, 1000, 2000];

/** Per-round, per-agent argument text accumulated from streamed tokens. */
interface RoundArguments {
  A: string;
  B: string;
}

/** Live snapshot of the in-flight token stream for a single agent. */
interface StreamingTokens {
  agent: "A" | "B";
  text: string;
  round: 1 | 2;
}

/**
 * Flattened, render-ready projection of every debate event observed so far.
 */
export interface DebateState {
  status: JobStatus;
  agentA: AgentResult | null;
  agentB: AgentResult | null;
  trigger: TriggerResult | null;
  attention: AttentionResult | null;
  argumentsRound1: RoundArguments;
  argumentsRound2: RoundArguments;
  streamingTokens: StreamingTokens | null;
  consensus: ConsensusResult | null;
  error: string | null;
  isConnected: boolean;
}

/** The public hook return type: the state plus a manual disconnect handle. */
export type UseDebateStreamReturn = DebateState & {
  disconnect: () => void;
};

/** Initial state used when the hook mounts or the `jobId` changes. */
function createInitialState(): DebateState {
  return {
    status: "queued",
    agentA: null,
    agentB: null,
    trigger: null,
    attention: null,
    argumentsRound1: { A: "", B: "" },
    argumentsRound2: { A: "", B: "" },
    streamingTokens: null,
    consensus: null,
    error: null,
    isConnected: false,
  };
}

/**
 * Reduce a single {@link DebateEvent} into a new {@link DebateState}, applying
 * all updates immutably so React can detect the change by reference.
 *
 * @param prev - the current state.
 * @param event - the parsed WebSocket event.
 * @returns the next state (a new object when the event is meaningful).
 */
function reduceEvent(prev: DebateState, event: DebateEvent): DebateState {
  switch (event.type) {
    case "agents_running":
      return { ...prev, status: "running" };

    case "agents_done":
      return {
        ...prev,
        agentA: event.agent_a,
        agentB: event.agent_b,
        status: "agents_done",
      };

    case "trigger_evaluated":
      return {
        ...prev,
        trigger: event.result,
        status: "trigger_evaluated",
      };

    case "attention_computed":
      return {
        ...prev,
        attention: event.result,
        status: "attention_computed",
      };

    case "argument_token": {
      const roundKey = event.round === 1 ? "argumentsRound1" : "argumentsRound2";
      const buffer = prev[roundKey];
      const updatedBuffer: RoundArguments = {
        ...buffer,
        [event.agent]: buffer[event.agent] + event.token,
      };
      const priorText =
        prev.streamingTokens && prev.streamingTokens.agent === event.agent
          ? prev.streamingTokens.text
          : "";
      return {
        ...prev,
        [roundKey]: updatedBuffer,
        streamingTokens: {
          agent: event.agent,
          text: priorText + event.token,
          round: event.round,
        },
        status: event.round === 1 ? "debate_round_1" : "debate_round_2",
      };
    }

    case "argument_done": {
      const roundKey = event.round === 1 ? "argumentsRound1" : "argumentsRound2";
      const buffer = prev[roundKey];
      const finalizedBuffer: RoundArguments = {
        ...buffer,
        [event.agent]: event.argument,
      };
      return {
        ...prev,
        [roundKey]: finalizedBuffer,
        streamingTokens: null,
        status: event.round === 1 ? "debate_round_1" : "debate_round_2",
      };
    }

    case "consensus_done":
      return {
        ...prev,
        consensus: event.result,
        status: "consensus_done",
      };

    case "error":
      return {
        ...prev,
        error: event.message,
        status: "failed",
      };

    default:
      return prev;
  }
}

/** Statuses after which the socket must not auto-reconnect. */
function isTerminalStatus(status: JobStatus): boolean {
  return status === "consensus_done" || status === "failed";
}

/** A minimal shape for the runtime-only `ping` keep-alive frame, which is not
 *  part of the typed {@link DebateEvent} union and is silently ignored. */
interface PingFrame {
  type: "ping";
}

/** Narrow an unknown parsed payload to the runtime `ping` keep-alive frame. */
function isPingFrame(value: unknown): value is PingFrame {
  return (
    typeof value === "object" &&
    value !== null &&
    (value as { type?: unknown }).type === "ping"
  );
}

/**
 * Subscribe to the debate WebSocket for `jobId` and expose the accumulated
 * {@link DebateState} together with a `disconnect` handle.
 *
 * @param jobId - the classification job identifier to stream.
 * @returns the live debate state and a manual disconnect function.
 */
export function useDebateStream(jobId: string): UseDebateStreamReturn {
  const [state, setState] = useState<DebateState>(createInitialState);

  const socketRef = useRef<WebSocket | null>(null);
  const retryCountRef = useRef<number>(0);
  const retryTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  /** When true, all reconnection and socket activity is permanently halted. */
  const manualCloseRef = useRef<boolean>(false);
  /** Mirrors the latest status so close handlers can read it synchronously. */
  const statusRef = useRef<JobStatus>("queued");

  /** Cancel any pending reconnection timer. */
  const clearRetryTimer = useCallback((): void => {
    if (retryTimerRef.current !== null) {
      clearTimeout(retryTimerRef.current);
      retryTimerRef.current = null;
    }
  }, []);

  /**
   * Permanently close the connection: stop reconnecting, clear timers, and
   * tear down the active socket. Safe to call multiple times.
   */
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
    setState((prev) => (prev.isConnected ? { ...prev, isConnected: false } : prev));
  }, [clearRetryTimer]);

  useEffect(() => {
    if (!jobId) {
      return undefined;
    }

    // Reset all per-job state when (re)subscribing.
    manualCloseRef.current = false;
    retryCountRef.current = 0;
    statusRef.current = "queued";
    setState(createInitialState());

    /** Open a socket and wire up its event handlers. */
    const connect = (): void => {
      if (manualCloseRef.current) {
        return;
      }

      const socket = new WebSocket(`${WS_URL}/debate/${jobId}`);
      socketRef.current = socket;

      socket.onopen = (): void => {
        retryCountRef.current = 0;
        setState((prev) => ({ ...prev, isConnected: true }));
      };

      socket.onmessage = (message: MessageEvent<string>): void => {
        let parsed: unknown;
        try {
          parsed = JSON.parse(message.data);
        } catch {
          // Ignore frames that are not valid JSON.
          return;
        }

        if (isPingFrame(parsed)) {
          return;
        }

        const event = parsed as DebateEvent;
        setState((prev) => {
          const next = reduceEvent(prev, event);
          statusRef.current = next.status;
          return next;
        });
      };

      socket.onerror = (): void => {
        // Errors are surfaced through the subsequent `onclose`; no state change
        // is made here to avoid masking a clean terminal close.
      };

      socket.onclose = (): void => {
        socketRef.current = null;
        setState((prev) =>
          prev.isConnected ? { ...prev, isConnected: false } : prev,
        );

        if (manualCloseRef.current || isTerminalStatus(statusRef.current)) {
          return;
        }

        if (retryCountRef.current < MAX_RETRIES) {
          const delay =
            BACKOFF_MS[retryCountRef.current] ??
            BACKOFF_MS[BACKOFF_MS.length - 1];
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
