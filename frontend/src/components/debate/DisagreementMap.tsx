"use client";

/**
 * DisagreementMap — the spatial-attention grid. Lays four HeatmapCanvas tiles
 * side by side (Agent A's Grad-CAM++, Agent B's attention rollout, their
 * disagreement map, and the source lesion with the contested region boxed) so a
 * clinician can read *where* the two minds diverged, not just that they did. A
 * single overlay toggle crossfades the contested bounding box on the source
 * tile, and a row of region-stat chips quantifies each agent's attention mass
 * per region. Tile layout is fixed-square so nothing reflows as images land.
 */

import { useState } from "react";

import type { AttentionResult } from "@/types/debate";
import HeatmapCanvas from "@/components/debate/HeatmapCanvas";
import { AGENT_A, AGENT_B, COLORS, ATTENTION_CAPTIONS } from "@/lib/constants";

interface DisagreementMapProps {
  attention: AttentionResult;
  sourceB64?: string | null;
}

/** One labelled tile: mono uppercase title, heatmap canvas, then a faint caption. */
function MapCell({
  title,
  caption,
  children,
}: {
  title: string;
  caption: string;
  children: React.ReactNode;
}): React.JSX.Element {
  return (
    <div className="flex flex-col gap-2">
      <div className="font-mono text-[10px] font-medium uppercase tracking-wider text-ink-soft">
        {title}
      </div>
      {children}
      <p className="text-xs text-ink-faint">{caption}</p>
    </div>
  );
}

/** A labelled cluster of region-attention chips for one agent. */
function RegionGroup({
  label,
  accent,
  stats,
}: {
  label: string;
  accent: string;
  stats: Record<string, number>;
}): React.JSX.Element {
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-1.5">
        <span className="h-2 w-2 rounded-full" style={{ backgroundColor: accent }} />
        <span className="text-[10px] font-medium uppercase tracking-wider text-ink-faint">
          {label}
        </span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {Object.entries(stats).map(([key, value]) => (
          <span
            key={key}
            className="inline-flex items-center gap-1.5 rounded-md border border-hairline bg-surface-alt px-2 py-1"
          >
            <span className="font-mono text-[10px] uppercase tracking-wider text-ink-faint">
              {key}
            </span>
            <span className="font-mono text-[11px] tabular text-ink">
              {value.toFixed(2)}
            </span>
          </span>
        ))}
      </div>
    </div>
  );
}

export default function DisagreementMap({
  attention,
  sourceB64,
}: DisagreementMapProps): React.JSX.Element {
  const [overlay, setOverlay] = useState(true);

  return (
    <section
      aria-label="Spatial attention analysis"
      className="rounded-2xl border border-hairline bg-surface p-6 shadow-panel animate-panel-enter"
    >
      {/* Header */}
      <header className="mb-5 flex items-center justify-between gap-4">
        <h3 className="font-display text-xl leading-tight text-ink">
          Spatial attention analysis
        </h3>
        <button
          type="button"
          onClick={() => setOverlay((v) => !v)}
          aria-pressed={overlay ? "true" : "false"}
          className="rounded-full border border-hairline px-3 py-1 font-mono text-[10px] font-medium uppercase tracking-wider text-ink-soft transition-colors hover:bg-surface-alt focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ink-faint"
        >
          {overlay ? "Show raw" : "Show overlay"}
        </button>
      </header>

      {/* Tile grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <MapCell title="Agent A · Grad-CAM++" caption={ATTENTION_CAPTIONS.a}>
          <HeatmapCanvas
            b64={attention.heatmap_a_b64}
            accent={AGENT_A.color}
            showOverlay={false}
            alt="Agent A Grad-CAM++"
          />
        </MapCell>

        <MapCell title="Agent B · Attn Rollout" caption={ATTENTION_CAPTIONS.b}>
          <HeatmapCanvas
            b64={attention.heatmap_b_b64}
            accent={AGENT_B.color}
            showOverlay={false}
            alt="Agent B Attention Rollout"
          />
        </MapCell>

        <MapCell title="Disagreement" caption={ATTENTION_CAPTIONS.disagreement}>
          <HeatmapCanvas
            b64={attention.disagreement_b64}
            accent={COLORS.warning}
            showOverlay={false}
            alt="Disagreement map"
          />
        </MapCell>

        <MapCell title="Source · contested" caption={ATTENTION_CAPTIONS.source}>
          <HeatmapCanvas
            b64={sourceB64 ?? attention.disagreement_b64}
            bbox={attention.bbox}
            accent={COLORS.danger}
            showOverlay={overlay}
            alt="Source lesion with contested region"
          />
        </MapCell>
      </div>

      {/* Region-stat chips */}
      <div className="mt-6 flex flex-col gap-4 border-t border-hairline pt-5 sm:flex-row sm:gap-10">
        <RegionGroup
          label="Agent A region"
          accent={AGENT_A.color}
          stats={attention.region_stats_a}
        />
        <RegionGroup
          label="Agent B region"
          accent={AGENT_B.color}
          stats={attention.region_stats_b}
        />
      </div>
    </section>
  );
}
