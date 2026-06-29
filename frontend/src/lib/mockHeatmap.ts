/**
 * Procedural saliency-map generator for mock mode.
 *
 * With no trained checkpoints the backend returns no heatmaps, so for a
 * convincing offline demo we synthesise plausible-looking attention overlays in
 * the browser: a dermoscopy-toned base with a few soft "hot" Gaussian blobs in a
 * JET-style colour ramp. Deterministic per `seed` so the same demo renders the
 * same image every time. Returns a PNG data URL, or "" on the server.
 */

interface Blob {
  /** Centre, in 0..1 of the image. */
  cx: number;
  cy: number;
  /** Radius, in 0..1 of the image. */
  r: number;
  /** Peak intensity 0..1. */
  peak: number;
}

interface HeatmapOptions {
  /** Hot regions to paint. */
  blobs: Blob[];
  /** Render the dermoscopy-toned lesion base beneath the heat (default true). */
  base?: boolean;
  /** Output size in px (square). Default 256. */
  size?: number;
}

/** A small JET-like colour ramp: blue → cyan → green → yellow → red. */
function jet(t: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, t));
  const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * x - 3)));
  const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * x - 2)));
  const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * x - 1)));
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

/**
 * Synthesise a heatmap overlay image and return it as a base64 PNG data URL.
 * Returns "" when called outside the browser (SSR).
 */
export function makeMockHeatmap(opts: HeatmapOptions): string {
  if (typeof document === "undefined") return "";

  const size = opts.size ?? 256;
  const withBase = opts.base ?? true;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";

  const img = ctx.createImageData(size, size);
  const data = img.data;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      const u = x / size;
      const v = y / size;

      // Dermoscopy-toned base: warm brown lesion fading to dark edges.
      let br = 26;
      let bg = 18;
      let bb = 22;
      if (withBase) {
        const dl = Math.hypot(u - 0.5, v - 0.5);
        const lesion = Math.max(0, 1 - dl * 1.9);
        br = 40 + lesion * 120;
        bg = 26 + lesion * 70;
        bb = 30 + lesion * 55;
      }

      // Accumulate Gaussian heat.
      let heat = 0;
      for (const blob of opts.blobs) {
        const d = Math.hypot(u - blob.cx, v - blob.cy);
        heat += blob.peak * Math.exp(-(d * d) / (2 * blob.r * blob.r));
      }
      heat = Math.max(0, Math.min(1, heat));

      const [hr, hg, hb] = jet(heat);
      // Alpha-composite the colour ramp over the base by heat strength.
      const a = heat * 0.85;
      data[i] = Math.round(hr * a + br * (1 - a));
      data[i + 1] = Math.round(hg * a + bg * (1 - a));
      data[i + 2] = Math.round(hb * a + bb * (1 - a));
      data[i + 3] = 255;
    }
  }

  ctx.putImageData(img, 0, 0);
  return canvas.toDataURL("image/png").split(",")[1] ?? "";
}

/** A neutral dermoscopy-toned lesion (no heat) for the "source image" panel. */
export function makeMockSource(size = 256): string {
  return makeMockHeatmap({ blobs: [], base: true, size });
}
