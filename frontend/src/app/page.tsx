"use client";

import { useCallback, useState } from "react";
import { useRouter } from "next/navigation";

import DropZone from "@/components/upload/DropZone";
import ImagePreview from "@/components/upload/ImagePreview";
import LoadingOrbit from "@/components/ui/LoadingOrbit";
import WebGLBackground from "@/components/debate/WebGLBackground";
import { uploadImage } from "@/lib/api";
import { fileToDataUrl, storeJobImage } from "@/lib/sessionImage";
import { ISIC_CLASSES } from "@/lib/constants";

/** What the pipeline does, in the user's terms. */
const CAPABILITIES: ReadonlyArray<{ title: string; body: string }> = [
  {
    title: "Adversarial debate",
    body: "A CNN and a Vision Transformer analyze the lesion independently — and argue when they disagree.",
  },
  {
    title: "Spatial evidence",
    body: "Grad-CAM++ and attention rollout reveal exactly where each agent looked, and where they conflict.",
  },
  {
    title: "Calibrated consensus",
    body: "A calibrated fusion head resolves the debate into a temperature-scaled, ECE-reported verdict.",
  },
];

export default function HomePage(): React.JSX.Element {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelected = useCallback(
    (selected: File): void => {
      setError(null);
      if (previewUrl !== null) URL.revokeObjectURL(previewUrl);
      setFile(selected);
      setPreviewUrl(URL.createObjectURL(selected));
    },
    [previewUrl],
  );

  const handleClear = useCallback((): void => {
    if (previewUrl !== null) URL.revokeObjectURL(previewUrl);
    setFile(null);
    setPreviewUrl(null);
    setError(null);
  }, [previewUrl]);

  const handleAnalyze = useCallback(async (): Promise<void> => {
    if (file === null || isUploading) return;
    setIsUploading(true);
    setError(null);
    try {
      // Capture the image for the debate page before uploading.
      let dataUrl: string | null = null;
      try {
        dataUrl = await fileToDataUrl(file);
      } catch {
        dataUrl = null;
      }
      const { job_id } = await uploadImage(file);
      if (dataUrl) storeJobImage(job_id, dataUrl);
      router.push(`/debate/${job_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to upload image.");
      setIsUploading(false);
    }
  }, [file, isUploading, router]);

  return (
    <main className="bg-theatre relative min-h-screen w-full overflow-hidden">
      <WebGLBackground mode="idle" />

      <div className="relative z-10 mx-auto flex min-h-screen w-full max-w-5xl flex-col px-6 py-8">
        {/* Wordmark */}
        <header className="flex items-center gap-2 select-none">
          <span className="grid h-8 w-8 place-items-center rounded-md border border-hairline bg-surface font-mono text-[10px] text-ink-faint shadow-panel">
            AV
          </span>
          <span className="text-[13px] font-semibold tracking-[0.28em] text-ink">
            ARGUS<span className="text-ink-faint"> VISION</span>
          </span>
        </header>

        {/* Hero */}
        <section className="flex flex-1 flex-col items-center justify-center py-12 text-center">
          <span className="animate-panel-enter text-[11px] font-medium uppercase tracking-[0.22em] text-ink-faint">
            Live adversarial diagnosis
          </span>
          <h1
            className="animate-panel-enter mt-4 font-display text-5xl leading-[1.05] text-ink sm:text-6xl"
            style={{ animationDelay: "60ms" }}
          >
            Two minds, arguing
            <br />
            toward one diagnosis.
          </h1>
          <p
            className="animate-panel-enter mt-5 max-w-xl text-base leading-relaxed text-ink-soft"
            style={{ animationDelay: "120ms" }}
          >
            Upload a dermoscopic skin-lesion image and watch a CNN and a Vision
            Transformer reason, disagree, and resolve to a calibrated verdict —
            in real time.
          </p>

          {/* Upload */}
          <div className="mt-10 w-full max-w-md">
            {file !== null && previewUrl !== null ? (
              <div className="flex flex-col items-center gap-6">
                <ImagePreview src={previewUrl} fileName={file.name} onClear={handleClear} />
                <button
                  type="button"
                  onClick={handleAnalyze}
                  disabled={isUploading}
                  className="inline-flex items-center justify-center gap-3 rounded-xl bg-agent-a px-8 py-3 text-sm font-semibold uppercase tracking-widest text-white shadow-glow-a transition-all hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isUploading ? (
                    <>
                      <LoadingOrbit size={20} />
                      <span>Uploading</span>
                    </>
                  ) : (
                    <span>Begin debate</span>
                  )}
                </button>
              </div>
            ) : (
              <DropZone onFileSelected={handleFileSelected} />
            )}
          </div>

          {error !== null && (
            <div
              role="alert"
              className="mt-6 w-full max-w-md rounded-xl border border-danger/40 bg-danger/5 px-4 py-3 font-mono text-sm text-danger"
            >
              {error}
            </div>
          )}

          {/* Differential */}
          <div className="mt-10 flex flex-wrap items-center justify-center gap-1.5">
            {ISIC_CLASSES.map((c) => (
              <span
                key={c.id}
                title={c.fullName}
                className="rounded-full border border-hairline bg-surface/70 px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-ink-faint backdrop-blur-sm"
              >
                {c.id}
              </span>
            ))}
          </div>
        </section>

        {/* Capabilities */}
        <section className="grid gap-4 border-t border-hairline pt-8 sm:grid-cols-3">
          {CAPABILITIES.map((cap, i) => (
            <div
              key={cap.title}
              className="animate-panel-enter rounded-2xl border border-hairline bg-surface/80 p-5 shadow-panel backdrop-blur-sm"
              style={{ animationDelay: `${i * 80}ms` }}
            >
              <div className="font-mono text-[11px] uppercase tracking-wider text-ink-faint">
                0{i + 1}
              </div>
              <h3 className="mt-1.5 font-display text-lg text-ink">{cap.title}</h3>
              <p className="mt-1 text-[13px] leading-relaxed text-ink-soft">{cap.body}</p>
            </div>
          ))}
        </section>

        <footer className="py-6 text-center font-mono text-[11px] text-ink-faint">
          Research prototype — not for clinical use.
        </footer>
      </div>
    </main>
  );
}
