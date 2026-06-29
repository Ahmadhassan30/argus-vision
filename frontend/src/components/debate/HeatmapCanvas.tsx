"use client";

/**
 * HeatmapCanvas — renders a base64 PNG (a Grad-CAM++/rollout overlay or the
 * source lesion) into a crisp square <canvas>, with an HTML bounding-box overlay
 * that can crossfade in and out (Show Overlay / Show Raw). When no image is
 * available it shows an intentional, on-brand placeholder rather than crashing.
 *
 * The bounding box is expressed in the image's natural pixel space, so it tracks
 * correctly whether the source is a 224px backend heatmap or a 256px mock.
 */

import { useEffect, useRef, useState } from "react";

import type { BoundingBox } from "@/types/debate";

/** Internal render resolution; CSS scales the canvas responsively. */
const RENDER_SIZE = 512;

interface HeatmapCanvasProps {
  /** Raw base64 PNG (no prefix) or a full data URL; null → placeholder. */
  b64: string | null;
  /** Contested-region box in the image's natural pixel space. */
  bbox?: BoundingBox | null;
  /** Accent colour for the bbox and placeholder. */
  accent: string;
  /** Whether to draw the bounding box overlay. */
  showOverlay?: boolean;
  alt: string;
  placeholderLabel?: string;
  className?: string;
}

function toDataUri(b64: string): string {
  return b64.startsWith("data:") ? b64 : `data:image/png;base64,${b64}`;
}

export default function HeatmapCanvas({
  b64,
  bbox = null,
  accent,
  showOverlay = true,
  alt,
  placeholderLabel = "Heatmap unavailable",
  className = "",
}: HeatmapCanvasProps): React.JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    setLoaded(false);
    setNatural(null);
    const canvas = canvasRef.current;
    if (!canvas || !b64) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let cancelled = false;
    const img = new Image();
    img.onload = (): void => {
      if (cancelled) return;
      ctx.clearRect(0, 0, RENDER_SIZE, RENDER_SIZE);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(img, 0, 0, RENDER_SIZE, RENDER_SIZE);
      setNatural({ w: img.naturalWidth || RENDER_SIZE, h: img.naturalHeight || RENDER_SIZE });
      setLoaded(true);
    };
    img.onerror = (): void => {
      if (!cancelled) setLoaded(false);
    };
    img.src = toDataUri(b64);
    return () => {
      cancelled = true;
    };
  }, [b64]);

  const box =
    bbox && natural
      ? {
          left: `${(bbox.x1 / natural.w) * 100}%`,
          top: `${(bbox.y1 / natural.h) * 100}%`,
          width: `${((bbox.x2 - bbox.x1) / natural.w) * 100}%`,
          height: `${((bbox.y2 - bbox.y1) / natural.h) * 100}%`,
        }
      : null;

  const showPlaceholder = !b64 || !loaded;

  return (
    <div
      className={[
        "relative aspect-square w-full overflow-hidden rounded-lg border border-hairline bg-surface-alt",
        className,
      ].join(" ")}
      role="img"
      aria-label={alt}
    >
      <canvas
        ref={canvasRef}
        width={RENDER_SIZE}
        height={RENDER_SIZE}
        className="block h-full w-full transition-opacity duration-300"
        style={{ opacity: showPlaceholder ? 0 : 1 }}
      />

      {/* Bounding-box overlay (crossfades with showOverlay). */}
      {box && (
        <div
          className="pointer-events-none absolute rounded-[3px] border-2 transition-opacity duration-300"
          style={{
            left: box.left,
            top: box.top,
            width: box.width,
            height: box.height,
            borderColor: accent,
            boxShadow: `0 0 0 1px rgba(255,255,255,0.6), 0 0 16px ${accent}55`,
            opacity: showOverlay && !showPlaceholder ? 1 : 0,
          }}
        >
          <span
            className="absolute -top-[18px] left-0 rounded-sm px-1 py-px font-mono text-[9px] font-medium uppercase tracking-wider text-white"
            style={{ backgroundColor: accent }}
          >
            contested
          </span>
        </div>
      )}

      {/* On-brand placeholder. */}
      {showPlaceholder && (
        <div
          className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-center"
          style={{
            background: `linear-gradient(150deg, ${accent}1f 0%, #ffffff 70%)`,
          }}
        >
          <svg
            width="26"
            height="26"
            viewBox="0 0 24 24"
            fill="none"
            aria-hidden
            style={{ color: accent }}
          >
            <path
              d="M9 3h6v6h6v6h-6v6H9v-6H3V9h6V3Z"
              fill="currentColor"
              opacity="0.85"
            />
          </svg>
          <span className="px-3 font-mono text-[10px] uppercase tracking-wider text-ink-faint">
            {placeholderLabel}
          </span>
        </div>
      )}
    </div>
  );
}
