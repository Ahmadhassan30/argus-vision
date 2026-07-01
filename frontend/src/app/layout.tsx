import type { Metadata } from "next";
import { Plus_Jakarta_Sans, JetBrains_Mono, Space_Grotesk } from "next/font/google";
import "./globals.css";

/**
 * Body / UI typeface (Plus Jakarta Sans) — premium, modern, and clinical-research grade.
 * Exposed as `--font-body` and wired to the Tailwind `font-body` family.
 */
const plusJakartaSans = Plus_Jakarta_Sans({
  subsets: ["latin"],
  display: "swap",
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-body",
});

/**
 * Data / numeric typeface (JetBrains Mono) — probabilities, gauges, hex codes.
 * Exposed as `--font-mono`.
 */
const jetBrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  display: "swap",
  weight: ["400", "500", "600"],
  variable: "--font-mono",
});

/**
 * Display typeface (Space Grotesk) — section headers, agent names, and title.
 * Futuristic, geometric, space-themed. Exposed as `--font-display`.
 */
const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  display: "swap",
  weight: ["400", "500", "600", "700"],
  variable: "--font-display",
});

/** Page metadata for the Argus Vision application. */
export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost"),
  title: "Argus Vision — Live Diagnostic Debate",
  description:
    "Watch two AI agents reason, disagree, and argue their way to a calibrated diagnosis on a dermoscopic skin-lesion image.",
  openGraph: {
    title: "Argus Vision — Live Diagnostic Debate",
    description:
      "Watch two AI agents reason, disagree, and argue their way to a calibrated diagnosis.",
    images: [{ url: "/og.png" }],
  },
};

/**
 * Root layout. Wires the three typefaces as CSS variables on `<html>` and
 * applies the bright "Luminous Clinical Theatre" base to `<body>`.
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
        inter.variable,
        jetBrainsMono.variable,
        spaceGrotesk.variable,
      ].join(" ")}
    >
      <body className="min-h-screen bg-canvas font-body text-ink antialiased">
        {children}
      </body>
    </html>
  );
}
