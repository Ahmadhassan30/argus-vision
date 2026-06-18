import type { Metadata } from "next";
import { Space_Grotesk, JetBrains_Mono, DM_Serif_Display } from "next/font/google";
import "./globals.css";

/**
 * Display typeface (Space Grotesk) exposed as the `--font-display` CSS
 * variable and consumed by the Tailwind `font-display` family.
 */
const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-display",
});

/**
 * Monospace typeface (JetBrains Mono) exposed as the `--font-mono` CSS
 * variable and consumed by the Tailwind `font-mono` family.
 */
const jetBrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-mono",
});

/**
 * Serif display typeface (DM Serif Display) exposed as the `--font-serif`
 * CSS variable and consumed by the Tailwind `font-serif` family.
 */
const dmSerifDisplay = DM_Serif_Display({
  subsets: ["latin"],
  display: "swap",
  weight: "400",
  variable: "--font-serif",
});

/** Page metadata for the Argus Vision application. */
export const metadata: Metadata = {
  title: "Argus Vision",
  description:
    "Adversarial multi-agent visual debate for uncertainty-aware medical image classification.",
  openGraph: {
    title: "Argus Vision",
    description:
      "Adversarial multi-agent visual debate for uncertainty-aware medical image classification.",
    images: [{ url: "/og.png" }],
  },
};

/**
 * Root layout for the application. Wires the three Google fonts as CSS
 * variables on the `<html>` element and applies the global dark theme to the
 * `<body>`.
 *
 * @param props.children - The routed page content to render.
 */
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}): React.JSX.Element {
  return (
    <html
      lang="en"
      className={[
        spaceGrotesk.variable,
        jetBrainsMono.variable,
        dmSerifDisplay.variable,
      ].join(" ")}
    >
      <body className="bg-argus-black text-white min-h-screen antialiased">
        {children}
      </body>
    </html>
  );
}
