/**
 * Hand the uploaded lesion image from the upload page to the debate page.
 *
 * The backend never serves the original image back, so we stash it as a data URL
 * in sessionStorage keyed by job id at upload time, and the debate page reads it
 * to show the real specimen alongside the live debate. Best-effort: storage
 * quota or SSR simply means no source thumbnail, never an error.
 */

const PREFIX = "argus:image:";

/** Read a File as a base64 data URL. */
export function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(typeof reader.result === "string" ? reader.result : "");
    reader.onerror = () => reject(new Error("Failed to read image"));
    reader.readAsDataURL(file);
  });
}

/** Store a job's source image data URL (best-effort). */
export function storeJobImage(jobId: string, dataUrl: string): void {
  if (typeof window === "undefined") return;
  try {
    window.sessionStorage.setItem(PREFIX + jobId, dataUrl);
  } catch {
    // Quota exceeded or storage unavailable — skip the source thumbnail.
  }
}

/** Load a job's source image data URL, or null if absent. */
export function loadJobImage(jobId: string): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.sessionStorage.getItem(PREFIX + jobId);
  } catch {
    return null;
  }
}
