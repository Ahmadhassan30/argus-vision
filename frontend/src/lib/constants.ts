/**
 * Static metadata for the Argus Vision frontend: the ISIC-8 class table,
 * agent brand colors, and risk-level color mapping. These values are the
 * single source of truth for labels and colors derived from the shared
 * project contract.
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
 * names and risk levels as specified in the shared contract.
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

/** Brand color for Agent A (EfficientNet-B4). */
export const AGENT_A_COLOR = "#3B7DD8";

/** Brand color for Agent B (ViT-B/16). */
export const AGENT_B_COLOR = "#D4A017";

/**
 * Maps a risk level to its display color: low -> consensus green,
 * medium -> warning amber, high -> danger red.
 */
export const RISK_COLORS: Record<RiskLevel, string> = {
  low: "#22C55E",
  medium: "#F59E0B",
  high: "#EF4444",
};

/**
 * Returns the class metadata for the given class id, or `undefined` if the
 * id is not one of the 8 known ISIC classes.
 *
 * @param id - The class identifier (e.g. "MEL").
 */
export function getClassMeta(id: string): ClassMeta | undefined {
  return ISIC_CLASSES.find((cls) => cls.id === id);
}

/**
 * Returns the risk color for the given class id, falling back to the muted
 * color when the id is unknown.
 *
 * @param id - The class identifier (e.g. "MEL").
 */
export function getRisk(id: string): string {
  const meta = getClassMeta(id);
  return meta ? RISK_COLORS[meta.risk] : "#6B7280";
}
