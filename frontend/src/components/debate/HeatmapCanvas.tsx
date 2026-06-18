"use client";

/**
 * HeatmapCanvas draws a 224x224 visualization onto an HTML <canvas>, scaled
 * responsively to its container. It optionally composites three layers:
 *
 *   1. A base layer — either a supplied original image (drawn to fill the
 *      canvas) or a flat dark backdrop when no original image is provided.
 *   2. A decoded base64 PNG heatmap drawn over the base at 40% opacity.
 *   3. A dashed, pulsing bounding box (in the danger color) marking a
 *      contested region, with coordinates scaled from the 224px model space
 *      to the canvas display size.
 *
 * All images are decoded asynchronously in effects (awaiting `onload`) before
 * being drawn, so the canvas only paints fully-decoded bitmaps.
 */

import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import type { BoundingBox } from "@/types/debate";

/** The native model-space resolution of every heatmap/image in pixels. */
const MODEL_SIZE = 224;

/** Opacity applied to the heatmap layer when compositing over the base. */
const HEATMAP_ALPHA = 0.4;

/** Danger color used for the contested-region bounding box. */
const BBOX_COLOR = "#EF4444";

/** Props for {@link HeatmapCanvas}. */
export interface HeatmapCanvasProps {
  /** Optional original image source (any drawable URL or data URI). */
  originalImageSrc: string | null;
  /** Optional raw base64 PNG (without the data-URI prefix) for the heatmap. */
  heatmapB64: string | null;
  /** Optional contested-region box, expressed in 224px model space. */
  bbox: BoundingBox | null;
}

/**
 * Loads an image source and resolves with the decoded {@link HTMLImageElement}.
 * Rejects if the image fails to load.
 *
 * @param src - The image source URL or data URI to decode.
 * @returns A promise resolving to the loaded image element.
 */
function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise<HTMLImageElement>((resolve, reject) => {
    const img = new Image();
    img.onload = (): void => resolve(img);
    img.onerror = (): void => reject(new Error("Failed to load image"));
    img.src = src;
  });
}

/**
 * Normalizes a raw base64 string into a PNG data URI. If the value already
 * carries a data-URI prefix it is returned unchanged.
 *
 * @param b64 - Raw base64 payload or an existing data URI.
 * @returns A `data:image/png;base64,...` URI.
 */
function toPngDataUri(b64: string): string {
  return b64.startsWith("data:") ? b64 : `data:image/png;base64,${b64}`;
}

/**
 * A responsive canvas that composites an optional original image, an optional
 * heatmap overlay, and an optional pulsing contested-region bounding box.
 *
 * @param props - The original image, heatmap base64, and bounding box.
 * @returns The rendered canvas (wrapped in a fade-in container).
 */
export default function HeatmapCanvas({
  originalImageSrc,
  heatmapB64,
  bbox,
}: HeatmapCanvasProps): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [baseImage, setBaseImage] = useState<HTMLImageElement | null>(null);
  const [heatmapImage, setHeatmapImage] = useState<HTMLImageElement | null>(
    null
  );

  // Decode the optional original/base image whenever its source changes.
  useEffect(() => {
    let cancelled = false;
    if (!originalImageSrc) {
      setBaseImage(null);
      return;
    }
    loadImage(originalImageSrc)
      .then((img) => {
        if (!cancelled) {
          setBaseImage(img);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setBaseImage(null);
        }
      });
    return (): void => {
      cancelled = true;
    };
  }, [originalImageSrc]);

  // Decode the optional heatmap PNG whenever its base64 payload changes.
  useEffect(() => {
    let cancelled = false;
    if (!heatmapB64) {
      setHeatmapImage(null);
      return;
    }
    loadImage(toPngDataUri(heatmapB64))
      .then((img) => {
        if (!cancelled) {
          setHeatmapImage(img);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setHeatmapImage(null);
        }
      });
    return (): void => {
      cancelled = true;
    };
  }, [heatmapB64]);

  // Composite the layers onto the canvas whenever any input changes.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Base layer.
    if (baseImage) {
      ctx.globalAlpha = 1;
      ctx.drawImage(baseImage, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#0A0A0F";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    // Heatmap overlay at reduced opacity.
    if (heatmapImage) {
      ctx.globalAlpha = HEATMAP_ALPHA;
      ctx.drawImage(heatmapImage, 0, 0, canvas.width, canvas.height);
      ctx.globalAlpha = 1;
    }
  }, [baseImage, heatmapImage]);

  // Geometry for the optional pulsing bounding-box overlay (in percentages so
  // it tracks the responsively-scaled canvas without re-measuring).
  const overlay =
    bbox !== null
      ? {
          left: `${(bbox.x1 / MODEL_SIZE) * 100}%`,
          top: `${(bbox.y1 / MODEL_SIZE) * 100}%`,
          width: `${((bbox.x2 - bbox.x1) / MODEL_SIZE) * 100}%`,
          height: `${((bbox.y2 - bbox.y1) / MODEL_SIZE) * 100}%`,
        }
      : null;

  return (
    <motion.div
      className="relative w-full overflow-hidden rounded-md border border-argus-border bg-argus-black"
      style={{ aspectRatio: "1 / 1" }}
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
    >
      <canvas
        ref={canvasRef}
        width={MODEL_SIZE}
        height={MODEL_SIZE}
        className="block h-full w-full"
      />
      {overlay !== null && (
        <motion.div
          className="pointer-events-none absolute rounded-sm border-2 border-dashed"
          style={{
            left: overlay.left,
            top: overlay.top,
            width: overlay.width,
            height: overlay.height,
            borderColor: BBOX_COLOR,
          }}
          animate={{
            opacity: [0.45, 1, 0.45],
            boxShadow: [
              `0 0 0px 0px ${BBOX_COLOR}00`,
              `0 0 14px 2px ${BBOX_COLOR}99`,
              `0 0 0px 0px ${BBOX_COLOR}00`,
            ],
          }}
          transition={{ duration: 1.8, ease: "easeInOut", repeat: Infinity }}
        />
      )}
    </motion.div>
  );
}
