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
    <main className="bg-theatre relative min-h-screen w-full overflow-x-hidden">
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
        <section className="flex flex-1 flex-col items-center justify-center py-16 text-center">
          <div className="animate-panel-enter flex items-center gap-2.5 rounded-full border border-agent-a/30 bg-agent-a/5 px-3 py-1 font-mono text-[9px] font-bold uppercase tracking-[0.25em] text-agent-a shadow-[0_0_15px_rgba(59,130,246,0.1)]">
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500"></span>
            </span>
            SYS_GATE // DEBATE SYSTEM ACTIVE
          </div>
          <h1
            className="animate-panel-enter mt-6 font-display text-6xl font-extrabold uppercase leading-[0.95] tracking-tighter sm:text-7xl md:text-8xl bg-gradient-to-b from-white via-slate-100 to-indigo-300 bg-clip-text text-transparent"
            style={{ animationDelay: "60ms" }}
          >
            Two Minds.
            <br />
            One Consensus.
          </h1>
          <p
            className="animate-panel-enter mt-6 max-w-2xl text-[14px] md:text-[16px] leading-relaxed text-ink-soft/90 font-mono"
            style={{ animationDelay: "120ms" }}
          >
            Deploy dual classifier agents (CNN + ViT) to analyze skin lesions. 
            When uncertainty peaks, they initiate spatial Grad-CAM++ disagreement mapping, 
            resolving features into a calibrated 23-dimensional verdict.
          </p>

          {/* Upload */}
          <div className="mt-12 w-full max-w-md">
            {file !== null && previewUrl !== null ? (
              <div className="flex flex-col items-center gap-6">
                <ImagePreview src={previewUrl} fileName={file.name} onClear={handleClear} />
                <button
                  type="button"
                  onClick={handleAnalyze}
                  disabled={isUploading}
                  className="inline-flex items-center justify-center gap-3 rounded-xl bg-agent-a px-8 py-3 text-sm font-semibold uppercase tracking-widest text-white shadow-glow-a transition-all hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60 font-mono"
                >
                  {isUploading ? (
                    <>
                      <LoadingOrbit size={20} />
                      <span>processing_data</span>
                    </>
                  ) : (
                    <span>run_pipeline</span>
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
              className="mt-6 w-full max-w-md rounded-xl border border-danger/40 bg-danger/5 px-4 py-3 font-mono text-xs text-danger uppercase tracking-wide"
            >
              {error}
            </div>
          )}

          {/* Differential */}
          <div className="mt-12 flex flex-wrap items-center justify-center gap-2">
            {ISIC_CLASSES.map((c) => (
              <span
                key={c.id}
                title={c.fullName}
                className="rounded-md border border-hairline bg-surface/30 px-3 py-1 font-mono text-[10px] uppercase tracking-wider text-ink-soft hover:text-white hover:border-agent-b transition-all backdrop-blur-md"
              >
                {c.id}
              </span>
            ))}
          </div>
        </section>

        {/* Capabilities */}
        <section className="grid gap-6 border-t border-hairline/50 pt-10 sm:grid-cols-3">
          {CAPABILITIES.map((cap, i) => (
            <div
              key={cap.title}
              className="group animate-panel-enter rounded-2xl border border-hairline bg-surface/50 p-6 shadow-panel backdrop-blur-md hover:border-agent-a/50 hover:shadow-[0_0_20px_rgba(59,130,246,0.1)] transition-all duration-300"
              style={{ animationDelay: `${i * 80}ms` }}
            >
              <div className="font-mono text-[11px] uppercase tracking-wider text-agent-a/80">
                NODE_0{i + 1}
              </div>
              <h3 className="mt-2 font-display text-xl font-bold uppercase tracking-tight text-white group-hover:text-agent-a transition-colors">{cap.title}</h3>
              <p className="mt-2 text-xs leading-relaxed text-ink-soft font-mono">{cap.body}</p>
            </div>
          ))}
        </section>

        <footer className="py-8 text-center font-mono text-[10px] uppercase tracking-widest text-ink-faint/70">
          SECURE PROTOCOL — RESEARCH FIXTURE ONLY
        </footer>
      </div>
    </main>
  );
}
