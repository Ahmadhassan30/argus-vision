import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        "argus-black": "#0A0A0F",
        "argus-surface": "#12121A",
        "argus-border": "#1E1E2E",
        "argus-agent-a": "#3B7DD8",
        "argus-agent-b": "#D4A017",
        "argus-consensus": "#22C55E",
        "argus-warning": "#F59E0B",
        "argus-danger": "#EF4444",
        "argus-muted": "#6B7280",
      },
      fontFamily: {
        display: ["var(--font-display)", "sans-serif"],
        mono: ["var(--font-mono)", "monospace"],
        serif: ["var(--font-serif)", "serif"],
      },
      keyframes: {
        "pulse-border": {
          "0%, 100%": { borderColor: "#3B7DD8" },
          "50%": { borderColor: "#D4A017" },
        },
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(8px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "stream-in": {
          "0%": { opacity: "0", transform: "translateX(-4px)" },
          "100%": { opacity: "1", transform: "translateX(0)" },
        },
        orbit: {
          "0%": { transform: "rotate(0deg) translateX(12px) rotate(0deg)" },
          "100%": {
            transform: "rotate(360deg) translateX(12px) rotate(-360deg)",
          },
        },
        blink: {
          "0%, 49%": { opacity: "1" },
          "50%, 100%": { opacity: "0" },
        },
        "bbox-glow": {
          "0%, 100%": {
            boxShadow: "0 0 0 0 rgba(245, 158, 11, 0.0)",
            borderColor: "#F59E0B",
          },
          "50%": {
            boxShadow: "0 0 16px 2px rgba(245, 158, 11, 0.55)",
            borderColor: "#EF4444",
          },
        },
      },
      animation: {
        "pulse-border": "pulse-border 2s ease-in-out infinite",
        "fade-in": "fade-in 0.4s ease-out",
        "stream-in": "stream-in 0.25s ease-out",
        orbit: "orbit 3s linear infinite",
        blink: "blink 1s step-end infinite",
        "bbox-glow": "bbox-glow 2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};

export default config;
