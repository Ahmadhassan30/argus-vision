"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import DropZone from "@/components/upload/DropZone";
import ImagePreview from "@/components/upload/ImagePreview";
import LoadingOrbit from "@/components/ui/LoadingOrbit";
import { uploadImage } from "@/lib/api";

/** The feature pills shown beneath the upload control. */
const FEATURE_PILLS: ReadonlyArray<string> = [
  "Adversarial Debate",
  "Spatial Evidence",
  "Calibrated Consensus",
];

/**
 * Landing and upload page. Lets the user select a dermoscopic image, preview
 * it, and submit it for analysis. On submission the image is uploaded and the
 * user is routed to the live debate stream for the created job.
 */
export default function HomePage(): React.JSX.Element {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  /** Handles a freshly selected file: stores it and builds a preview URL. */
  const handleFileSelected = useCallback(
    (selected: File): void => {
      setError(null);
      if (previewUrl !== null) {
        URL.revokeObjectURL(previewUrl);
      }
      setFile(selected);
      setPreviewUrl(URL.createObjectURL(selected));
    },
    [previewUrl],
  );

  /** Clears the current selection and revokes its preview URL. */
  const handleClear = useCallback((): void => {
    if (previewUrl !== null) {
      URL.revokeObjectURL(previewUrl);
    }
    setFile(null);
    setPreviewUrl(null);
    setError(null);
  }, [previewUrl]);

  /** Uploads the selected file and navigates to the debate stream page. */
  const handleAnalyze = useCallback(async (): Promise<void> => {
    if (file === null || isUploading) {
      return;
    }
    setIsUploading(true);
    setError(null);
    try {
      const response = await uploadImage(file);
      router.push(`/debate/${response.job_id}`);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Failed to upload image.";
      setError(message);
      setIsUploading(false);
    }
  }, [file, isUploading, router]);

  return (
    <main className="bg-argus-grid flex min-h-screen w-full flex-col items-center justify-center px-6 py-16">
      <header className="mb-12 select-none">
        <h1 className="font-display text-lg font-bold tracking-[0.35em]">
          ARGUS<span className="text-argus-muted"> VISION</span>
        </h1>
      </header>

      <section className="flex w-full max-w-2xl flex-col items-center text-center">
        <h2 className="font-serif text-5xl leading-tight text-white sm:text-6xl">
          Two agents. One truth.
        </h2>
        <p className="mt-5 max-w-xl font-display text-base text-argus-muted sm:text-lg">
          Upload a dermoscopic image and watch two AI agents debate its
          diagnosis in real time.
        </p>

        <div className="mt-12 w-full">
          <AnimatePresence mode="wait">
            {file !== null && previewUrl !== null ? (
              <motion.div
                key="preview"
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.96 }}
                transition={{ duration: 0.3, ease: "easeOut" }}
                className="flex flex-col items-center gap-6"
              >
                <ImagePreview
                  src={previewUrl}
                  fileName={file.name}
                  onClear={handleClear}
                />
                <button
                  type="button"
                  onClick={handleAnalyze}
                  disabled={isUploading}
                  className="font-display inline-flex items-center justify-center gap-3 rounded-lg border border-argus-agent-a bg-argus-agent-a/10 px-8 py-3 text-sm font-semibold uppercase tracking-widest text-white transition-colors hover:bg-argus-agent-a/20 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isUploading ? (
                    <>
                      <LoadingOrbit size={20} />
                      <span>Uploading</span>
                    </>
                  ) : (
                    <span>Analyze</span>
                  )}
                </button>
              </motion.div>
            ) : (
              <motion.div
                key="dropzone"
                initial={{ opacity: 0, scale: 0.96 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.96 }}
                transition={{ duration: 0.3, ease: "easeOut" }}
              >
                <DropZone onFileSelected={handleFileSelected} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {error !== null && (
          <div
            role="alert"
            className="mt-6 w-full rounded-lg border border-argus-danger/50 bg-argus-danger/10 px-4 py-3 font-mono text-sm text-argus-danger"
          >
            {error}
          </div>
        )}

        <ul className="mt-12 flex flex-wrap items-center justify-center gap-3">
          {FEATURE_PILLS.map((pill) => (
            <li
              key={pill}
              className="font-display rounded-full border border-argus-border bg-argus-surface px-4 py-2 text-xs uppercase tracking-wider text-argus-muted"
            >
              {pill}
            </li>
          ))}
        </ul>
      </section>
    </main>
  );
}
