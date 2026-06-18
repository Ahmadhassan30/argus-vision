"use client";

/**
 * useJobPoller — an HTTP polling fallback for when the debate WebSocket is
 * unavailable. While `enabled`, it polls the lightweight status endpoint every
 * two seconds; once the job reaches a terminal status it fetches the full
 * {@link JobResult} exactly once and stops the interval.
 */

import { useEffect, useRef, useState } from "react";

import { getJob, getJobStatus } from "@/lib/api";
import type { JobResult, JobStatus } from "@/types/debate";

/** Polling cadence for the status endpoint, in milliseconds. */
const POLL_INTERVAL_MS = 2000;

/** The hook's return shape: the latest status and the full job once resolved. */
export interface UseJobPollerReturn {
  status: JobStatus | null;
  job: JobResult | null;
}

/** Statuses at which polling should stop and the full job be fetched. */
function isTerminalStatus(status: JobStatus): boolean {
  return status === "consensus_done" || status === "failed";
}

/**
 * Poll a classification job's status as a fallback when the WebSocket stream
 * cannot be used.
 *
 * @param jobId - the classification job identifier to poll.
 * @param enabled - when false, no polling occurs and any active interval is
 *   torn down immediately.
 * @returns the latest known status and the full job result once terminal.
 */
export function useJobPoller(jobId: string, enabled: boolean): UseJobPollerReturn {
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [job, setJob] = useState<JobResult | null>(null);

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  /** Guards against issuing more than one full-job fetch per terminal job. */
  const fullJobFetchedRef = useRef<boolean>(false);
  /** True while the effect that owns the current interval is still mounted. */
  const activeRef = useRef<boolean>(false);

  useEffect(() => {
    if (!enabled || !jobId) {
      return undefined;
    }

    activeRef.current = true;
    fullJobFetchedRef.current = false;

    /** Stop the polling interval if one is running. */
    const stopInterval = (): void => {
      if (intervalRef.current !== null) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };

    /** Fetch the full job once, guarding against duplicate requests. */
    const fetchFullJob = async (): Promise<void> => {
      if (fullJobFetchedRef.current) {
        return;
      }
      fullJobFetchedRef.current = true;
      try {
        const result = await getJob(jobId);
        if (activeRef.current) {
          setJob(result);
          setStatus(result.status);
        }
      } catch {
        // Allow a later poll tick to retry the full-job fetch on failure.
        fullJobFetchedRef.current = false;
      }
    };

    /** Poll the status endpoint and react to terminal transitions. */
    const poll = async (): Promise<void> => {
      try {
        const { status: next } = await getJobStatus(jobId);
        if (!activeRef.current) {
          return;
        }
        setStatus(next);
        if (isTerminalStatus(next)) {
          stopInterval();
          await fetchFullJob();
        }
      } catch {
        // Transient errors are ignored; the next interval tick retries.
      }
    };

    // Kick off an immediate poll, then continue on the fixed interval.
    void poll();
    intervalRef.current = setInterval(() => {
      void poll();
    }, POLL_INTERVAL_MS);

    return (): void => {
      activeRef.current = false;
      stopInterval();
    };
  }, [jobId, enabled]);

  return { status, job };
}
