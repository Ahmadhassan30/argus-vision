/**
 * Static metadata for the Argus Vision frontend: the ISIC-8 class table, agent
 * identities, and the "Luminous Clinical Theatre" colour tokens needed in
 * JavaScript (canvas / WebGL) where Tailwind classes and CSS variables can't
 * reach. These values are the single source of truth for labels and the colours
 * derived from the shared project contract; they mirror tailwind.config.ts and
 * globals.css exactly.
 */

/** The risk severity associated with an ISIC class. */
export type RiskLevel = "low" | "medium" | "high";

/** Display metadata for a single ISIC class. */
export interface ClassMeta {
  id: string;
  label: string;
  fullName: string;
  risk: RiskLevel;
}

/**
 * The 8 ISIC classes in their exact contract order (index 0..7), with full
 * names and risk levels. Order matches backend `CLASS_NAMES`.
 */
export const ISIC_CLASSES: ReadonlyArray<ClassMeta> = [
  { id: "MEL", label: "MEL", fullName: "Melanoma", risk: "high" },
  { id: "NV", label: "NV", fullName: "Melanocytic Nevus", risk: "low" },
  { id: "BCC", label: "BCC", fullName: "Basal Cell Carcinoma", risk: "high" },
  { id: "AK", label: "AK", fullName: "Actinic Keratosis", risk: "medium" },
  { id: "BKL", label: "BKL", fullName: "Benign Keratosis", risk: "low" },
  { id: "DF", label: "DF", fullName: "Dermatofibroma", risk: "low" },
  { id: "VASC", label: "VASC", fullName: "Vascular Lesion", risk: "medium" },
  { id: "SCC", label: "SCC", fullName: "Squamous Cell Carcinoma", risk: "high" },
] as const;

/** Canonical class id order (index 0..7), matching the backend. */
export const CLASS_ORDER: ReadonlyArray<string> = ISIC_CLASSES.map((c) => c.id);

/** The two agent identifiers used throughout the debate UI. */
export type AgentId = "A" | "B";

/** Full identity + voice metadata for one debating agent. */
export interface AgentMeta {
  id: AgentId;
  /** Uppercase eyebrow label, e.g. "AGENT A". */
  label: string;
  /** Architecture / model name shown in display serif, e.g. "EfficientNet-B4". */
  name: string;
  /** One-line descriptor of the architecture's reasoning style. */
  descriptor: string;
  /** Hex accent colour for this agent. */
  color: string;
  /** Soft tint of the accent (for fills / highlights). */
  tint: string;
}

/** Agent A — EfficientNet-B4 CNN. Deep royal blue. */
export const AGENT_A: AgentMeta = {
  id: "A",
  label: "AGENT A",
  name: "EfficientNet-B4",
  descriptor: "CNN · Structural",
  color: "#2563EB",
  tint: "rgba(37, 99, 235, 0.10)",
};

/** Agent B — ViT-B/16 Transformer. Electric violet. */
export const AGENT_B: AgentMeta = {
  id: "B",
  label: "AGENT B",
  name: "ViT-B/16",
  descriptor: "Transformer · Global",
  color: "#7C3AED",
  tint: "rgba(124, 58, 237, 0.10)",
};

/** Lookup an agent's metadata by id. */
export const AGENTS: Record<AgentId, AgentMeta> = { A: AGENT_A, B: AGENT_B };

/** Backwards-compatible scalar exports for the agent accent colours. */
export const AGENT_A_COLOR = AGENT_A.color;
export const AGENT_B_COLOR = AGENT_B.color;

/** Outcome / status accent colours (mirror of the design tokens). */
export const COLORS = {
  canvas: "#F7F8FC",
  surface: "#FFFFFF",
  surfaceAlt: "#EFF1F8",
  agentA: "#2563EB",
  agentB: "#7C3AED",
  consensus: "#059669",
  warning: "#D97706",
  danger: "#DC2626",
  ink: "#0F172A",
  inkSoft: "#475569",
  inkFaint: "#94A3B8",
  hairline: "#E2E8F0",
} as const;

/**
 * Maps a risk level to its display colour: low → emerald, medium → amber,
 * high → red.
 */
export const RISK_COLORS: Record<RiskLevel, string> = {
  low: "#059669",
  medium: "#D97706",
  high: "#DC2626",
};

/** Human-readable risk wording for badges and tooltips. */
export const RISK_LABELS: Record<RiskLevel, string> = {
  low: "Low concern",
  medium: "Monitor",
  high: "Urgent",
};

/**
 * Returns the class metadata for the given class id, or `undefined` if the id
 * is not one of the 8 known ISIC classes.
 */
export function getClassMeta(id: string): ClassMeta | undefined {
  return ISIC_CLASSES.find((cls) => cls.id === id);
}

/** Returns the full diagnostic name for a class id, falling back to the id. */
export function getClassName(id: string): string {
  return getClassMeta(id)?.fullName ?? id;
}

/**
 * Returns the risk colour for the given class id, falling back to the muted
 * colour when the id is unknown.
 */
export function getRisk(id: string): string {
  const meta = getClassMeta(id);
  return meta ? RISK_COLORS[meta.risk] : COLORS.inkFaint;
}

/** Captions describing what each spatial-attention technique reveals. */
export const ATTENTION_CAPTIONS = {
  a: "Grad-CAM++ — gradient-weighted activations highlight the local texture and edge evidence Agent A weighed most.",
  b: "Attention Rollout — accumulated self-attention shows the global structure Agent B attended to across patches.",
  disagreement:
    "Disagreement Map — where the two agents looked at different things; the brighter the region, the sharper the conflict.",
  source: "Source — the dermoscopic lesion, with the contested region boxed.",
} as const;
