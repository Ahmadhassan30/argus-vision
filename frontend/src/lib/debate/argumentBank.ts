/**
 * Curated dermatological argument bank for retrieval-based debate text.
 *
 * Each fragment is a short, in-character line tagged by speaker architecture,
 * rhetorical move, and the visual-evidence theme it invokes. Placeholders are
 * filled at turn time with the actual classes and the opponent's read, so every
 * line is grounded in the live debate and references the other agent. Retrieval
 * (see retrieval.ts) embeds the turn's context and ranks fragments by semantic
 * similarity, avoiding recent repeats — so the conversation stays on-topic,
 * varied, and human, with no LLM in the loop.
 *
 * Placeholders: {self} {selfCode} {opp} {oppCode} {selfCue} {oppCue} {oppArch}
 */

import type { Move } from "@/lib/debate/beliefs";

export type EvidenceTag =
  | "texture"
  | "border"
  | "pigment"
  | "network"
  | "vascular"
  | "symmetry"
  | "global"
  | "color"
  | "structure"
  | "general";

export interface ArgFragment {
  id: string;
  /** Which architecture voice this line fits ("any" = either). */
  agent: "A" | "B" | "any";
  move: Move;
  evidence: EvidenceTag;
  /** Line with placeholders, filled per turn. */
  text: string;
}

/** Hallmark dermoscopy cue per class, injected as {selfCue} / {oppCue}. */
export const CLASS_CUE: Record<string, string> = {
  MEL: "asymmetric, atypical pigment network with an abrupt border cut-off",
  NV: "regular, symmetric reticular network fading evenly to the edge",
  BCC: "arborizing telangiectatic vessels and blue-grey ovoid nests",
  AK: "strawberry-pattern erythema with follicular plugging",
  BKL: "sharply demarcated milia-like cysts and comedo-like openings",
  DF: "central white scar-like patch ringed by a delicate pigment band",
  VASC: "well-defined red-purple lacunae separated by pale septa",
  SCC: "keratin pearls and looped, glomerular vessels over a scaly base",
};

/** Opponent-architecture phrase, injected as {oppArch} (from the speaker's POV). */
export const OPP_ARCH: Record<"A" | "B", string> = {
  // Speaker A's opponent is B (global/attention); speaker B's opponent is A (local/texture).
  A: "the global attention read",
  B: "the local texture read",
};

export const ARGUMENT_BANK: ReadonlyArray<ArgFragment> = [
  // ---- OPEN ---------------------------------------------------------------
  { id: "o-a1", agent: "A", move: "open", evidence: "texture", text: "Up close the surface tells the story — I'm reading {selfCue}, and that puts me on {self}." },
  { id: "o-a2", agent: "A", move: "open", evidence: "border", text: "The lateral margin shows {selfCue}; my call is {self} ({selfCode})." },
  { id: "o-a3", agent: "A", move: "open", evidence: "pigment", text: "Pigment detail is what I trust here — {selfCue} — and it reads as {self}." },
  { id: "o-b1", agent: "B", move: "open", evidence: "global", text: "Taking the lesion as a whole, the architecture reads as {selfCue}. I land on {self}." },
  { id: "o-b2", agent: "B", move: "open", evidence: "symmetry", text: "Across the field the structure is {selfCue}; at scale that's {self} to me." },
  { id: "o-b3", agent: "B", move: "open", evidence: "structure", text: "The patches relate to one another as {selfCue}, so my leading read is {self}." },
  { id: "o-x1", agent: "any", move: "open", evidence: "general", text: "My leading read is {self} — the evidence I weigh most points squarely there." },

  // ---- PRESS (hold ground) ------------------------------------------------
  { id: "p-a1", agent: "A", move: "press", evidence: "texture", text: "I hear you, but the fine detail hasn't changed: {selfCue} is still right there, and it keeps me on {self}." },
  { id: "p-a2", agent: "A", move: "press", evidence: "border", text: "Step back all you like — the border evidence, {selfCue}, doesn't soften. {self} stands." },
  { id: "p-b1", agent: "B", move: "press", evidence: "global", text: "Zooming out doesn't dissolve it — the global pattern of {selfCue} holds, so I stay with {self}." },
  { id: "p-b2", agent: "B", move: "press", evidence: "symmetry", text: "The overall symmetry argues the same thing it did a moment ago; I'm still reading {self}." },
  { id: "p-x1", agent: "any", move: "press", evidence: "general", text: "Nothing you've shown moves my strongest evidence — {self} still fits best." },
  { id: "p-x2", agent: "any", move: "press", evidence: "general", text: "I'll hold {self}; {oppArch} hasn't accounted for what I'm seeing." },

  // ---- REBUT (attack opponent's read) -------------------------------------
  { id: "r-a1", agent: "A", move: "rebut", evidence: "texture", text: "You're reasoning from the whole and missing the grain — if this were {opp} I'd expect a cleaner local texture, but I see {selfCue}." },
  { id: "r-a2", agent: "A", move: "rebut", evidence: "vascular", text: "Your {opp} doesn't explain the vessels up close. {selfCue} is the detail {oppArch} keeps skipping." },
  { id: "r-b1", agent: "B", move: "rebut", evidence: "global", text: "You're fixated on one patch. At full scale {opp} would organize differently; what I see is {selfCue}, not that." },
  { id: "r-b2", agent: "B", move: "rebut", evidence: "structure", text: "{oppCode} reads {opp} from a local fragment, but the lesion as a whole shows {selfCue} — that's the piece the close-up loses." },
  { id: "r-x1", agent: "any", move: "rebut", evidence: "general", text: "Your read of {opp} can't account for {selfCue} — that's exactly the evidence your account skips." },
  { id: "r-x2", agent: "any", move: "rebut", evidence: "general", text: "If it were truly {opp}, where is the {oppCue} you'd need? I don't see it carrying the lesion." },

  // ---- SOFTEN (partial shift) ---------------------------------------------
  { id: "s-a1", agent: "A", move: "soften", evidence: "texture", text: "Fair point — there's something to {opp}. I'll grant the {oppCue} you're pointing at, though {selfCue} still tilts me toward {self}." },
  { id: "s-b1", agent: "B", move: "soften", evidence: "global", text: "I'll concede the {oppCue} is real; it pulls me a little toward {opp}, even if the overall structure still favors {self}." },
  { id: "s-x1", agent: "any", move: "soften", evidence: "general", text: "That's a stronger case than I first allowed — I'm moving toward {opp}, but not all the way." },
  { id: "s-x2", agent: "any", move: "soften", evidence: "general", text: "Point taken. The {oppCue} deserves more weight than I gave it; my read is loosening from {self}." },

  // ---- CONCEDE (yield to opponent) ----------------------------------------
  { id: "c-a1", agent: "A", move: "concede", evidence: "general", text: "You've convinced me — the {oppCue} you keep returning to outweighs my texture read. I'll move to {opp}." },
  { id: "c-b1", agent: "B", move: "concede", evidence: "general", text: "Alright — the local evidence for {opp} is more decisive than my global impression. I'm with you on {opp}." },
  { id: "c-x1", agent: "any", move: "concede", evidence: "general", text: "I was overweighting my own signal; the case for {opp} is the better one. Conceding to {oppCode}." },

  // ---- AGREE (closing) ----------------------------------------------------
  { id: "a-x1", agent: "any", move: "agree", evidence: "general", text: "Then we're aligned — {self} it is, for the same reasons." },
  { id: "a-x2", agent: "any", move: "agree", evidence: "general", text: "Good — we've converged on {self}. The detail and the whole finally line up." },
  { id: "a-a1", agent: "A", move: "agree", evidence: "general", text: "We're reading the same lesion now: {self}." },
  { id: "a-b1", agent: "B", move: "agree", evidence: "general", text: "Agreed. Structure and texture both say {self}." },
];
