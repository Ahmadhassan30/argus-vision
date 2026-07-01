import type { Config } from "tailwindcss";

/**
 * Argus Vision — "Luminous Clinical Theatre" design system.
 *
 * A bright, high-contrast, scientifically precise palette: paper-white surfaces
 * lit like an operating theatre at noon, with electric agent accents that signal
 * live computational activity. The literal hex values live here (so Tailwind's
 * opacity modifiers such as `bg-agent-a/15` work natively); the same values are
 * mirrored as CSS custom properties in `globals.css` for raw-CSS needs and in
 * `lib/constants.ts` for canvas / WebGL use.
 */
const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Surfaces — the operating table.
        canvas: "var(--bg-canvas)",
        surface: "var(--bg-surface)",
        "surface-alt": "var(--bg-surface-alt)",
        // Agent + outcome accents.
        "agent-a": "var(--accent-agent-a)",
        "agent-b": "var(--accent-agent-b)",
        consensus: "var(--accent-consensus)",
        warning: "var(--accent-warning)",
        danger: "var(--accent-danger)",
        // Text + structure.
        ink: "var(--text-primary)",
        "ink-soft": "var(--text-secondary)",
        "ink-faint": "var(--text-muted)",
        hairline: "var(--border-subtle)",
      },
      fontFamily: {
        display: ["var(--font-display)", "Georgia", "serif"],
        body: ["var(--font-body)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "monospace"],
      },
      fontSize: {
        "2xs": ["11px", { lineHeight: "1.4" }],
        xs: ["13px", { lineHeight: "1.5" }],
        base: ["15px", { lineHeight: "1.6" }],
        lg: ["18px", { lineHeight: "1.5" }],
        xl: ["22px", { lineHeight: "1.3" }],
        "2xl": ["28px", { lineHeight: "1.2" }],
        "3xl": ["38px", { lineHeight: "1.1" }],
      },
      boxShadow: {
        "glow-a":
          "var(--glow-agent-a)",
        "glow-b":
          "var(--glow-agent-b)",
        "glow-consensus":
          "var(--glow-consensus)",
        "glow-warning":
          "0 0 0 3px rgba(217, 119, 6, 0.2), 0 4px 24px rgba(217, 119, 6, 0.18)",
        panel: "0 1px 3px rgba(0, 0, 0, 0.3), 0 1px 2px rgba(0, 0, 0, 0.2)",
        "panel-lg":
          "0 8px 32px rgba(0, 0, 0, 0.4), 0 2px 8px rgba(0, 0, 0, 0.2)",
      },
      keyframes: {
        // Status / thinking pulses.
        "pulse-dot": {
          "0%, 100%": { transform: "scale(1)", opacity: "1" },
          "50%": { transform: "scale(1.4)", opacity: "0.4" },
        },
        blink: { "50%": { opacity: "0" } },
        // Panel entrance.
        "panel-enter": {
          "0%": { opacity: "0", transform: "translateY(16px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        // Consensus reveal slides up further.
        "rise-in": {
          "0%": { opacity: "0", transform: "translateY(30px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        // Skeleton shimmer.
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        // Green resolution band sweeping the arena.
        sweep: {
          "0%": { transform: "translateX(-120%)", opacity: "0" },
          "30%": { opacity: "1" },
          "100%": { transform: "translateX(120%)", opacity: "0" },
        },
        // Three-dot thinking loader.
        "dot-bounce": {
          "0%, 80%, 100%": { transform: "translateY(0)", opacity: "0.35" },
          "40%": { transform: "translateY(-3px)", opacity: "1" },
        },
        // Amber flash on the VS badge when the trigger fires.
        "amber-flash": {
          "0%": { boxShadow: "0 0 0 0 rgba(217, 119, 6, 0.6)" },
          "100%": { boxShadow: "0 0 0 14px rgba(217, 119, 6, 0)" },
        },
        // Slow expand/contract border for the live "thinking" card.
        "breathe-a": {
          "0%, 100%": { boxShadow: "0 0 0 0 rgba(37, 99, 235, 0.0)" },
          "50%": { boxShadow: "0 0 0 4px rgba(37, 99, 235, 0.10)" },
        },
        "breathe-b": {
          "0%, 100%": { boxShadow: "0 0 0 0 rgba(124, 58, 237, 0.0)" },
          "50%": { boxShadow: "0 0 0 4px rgba(124, 58, 237, 0.10)" },
        },
      },
      animation: {
        "pulse-dot": "pulse-dot 1.2s ease-in-out infinite",
        "pulse-dot-fast": "pulse-dot 0.7s ease-in-out infinite",
        blink: "blink 1s step-start infinite",
        "panel-enter": "panel-enter 400ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "rise-in": "rise-in 500ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        shimmer: "shimmer 1.6s linear infinite",
        sweep: "sweep 850ms cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "dot-bounce": "dot-bounce 1.2s ease-in-out infinite",
        "amber-flash": "amber-flash 600ms ease-out",
        "breathe-a": "breathe-a 2s ease-in-out infinite",
        "breathe-b": "breathe-b 2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
