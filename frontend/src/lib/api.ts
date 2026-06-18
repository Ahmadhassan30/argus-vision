/**
 * HTTP client for the Argus Vision backend API. All requests are issued
 * against `NEXT_PUBLIC_API_URL` (defaulting to the nginx-exposed `/api`
 * prefix). Responses are typed against the shared schema mirrors.
 */
import type { JobResult, JobStatus } from "../types/debate";

/** Base URL for the backend API; nginx strips the `/api` prefix. */
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost/api";

/** Shape of the JSON body returned by `POST /classify`. */
interface ClassifyResponse {
  job_id: string;
  status: "queued";
  estimated_seconds: number;
}

/** Shape of the JSON body returned by `GET /jobs/{id}/status`. */
interface JobStatusResponse {
  status: JobStatus;
}

/** Shape of an error body returned by the backend (FastAPI `detail`). */
interface ErrorResponse {
  detail?: string;
}

/**
 * Extracts a human-readable error message from a failed response, falling
 * back to the HTTP status text when no `detail` field is present.
 *
 * @param res - The failed fetch Response.
 */
async function extractError(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as ErrorResponse;
    if (typeof body.detail === "string" && body.detail.length > 0) {
      return body.detail;
    }
  } catch {
    // Body was not JSON; fall through to status text.
  }
  return res.statusText || `Request failed with status ${res.status}`;
}

/**
 * Uploads an image for classification.
 *
 * @param file - The image file to classify.
 * @returns The created job's id.
 * @throws Error with the backend detail message when the request fails.
 */
export async function uploadImage(file: File): Promise<{ job_id: string }> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/classify`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    throw new Error(await extractError(res));
  }

  const data = (await res.json()) as ClassifyResponse;
  return { job_id: data.job_id };
}

/**
 * Fetches the full accumulated result for a job.
 *
 * @param jobId - The job identifier.
 * @returns The full JobResult.
 * @throws Error with the backend detail message when the request fails.
 */
export async function getJob(jobId: string): Promise<JobResult> {
  const res = await fetch(`${API_URL}/jobs/${jobId}`);

  if (!res.ok) {
    throw new Error(await extractError(res));
  }

  return (await res.json()) as JobResult;
}

/**
 * Fetches only the current status of a job.
 *
 * @param jobId - The job identifier.
 * @returns An object containing the job status.
 * @throws Error with the backend detail message when the request fails.
 */
export async function getJobStatus(
  jobId: string,
): Promise<{ status: JobStatus }> {
  const res = await fetch(`${API_URL}/jobs/${jobId}/status`);

  if (!res.ok) {
    throw new Error(await extractError(res));
  }

  return (await res.json()) as JobStatusResponse;
}

/**
 * Deletes a job and its associated state.
 *
 * @param jobId - The job identifier.
 * @throws Error with the backend detail message when the request fails.
 */
export async function deleteJob(jobId: string): Promise<void> {
  const res = await fetch(`${API_URL}/jobs/${jobId}`, {
    method: "DELETE",
  });

  if (!res.ok) {
    throw new Error(await extractError(res));
  }
}
