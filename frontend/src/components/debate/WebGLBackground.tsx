"use client";

/**
 * WebGLBackground — the ambient particle field that lives behind the entire
 * debate UI. It mounts a fixed, fullscreen, non-interactive <canvas> and hands
 * it to the vanilla-WebGL particle library, which drifts ~400 soft particles in
 * the two agent colours. The field is created once on mount (and torn down on
 * unmount), while changes to `mode` are pushed to the live field so the motion
 * can react to the debate — speeding up when divergence fires, slowing and
 * shifting to emerald on consensus. Reduced-motion handling lives in the lib.
 */

import { useEffect, useRef } from "react";

import { createParticleField, type FieldMode, type ParticleField } from "@/lib/webgl-particles";

interface WebGLBackgroundProps {
  /** Drives the field's behaviour: "idle" | "debate" | "consensus". */
  mode: FieldMode;
}

export default function WebGLBackground({ mode }: WebGLBackgroundProps): React.JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fieldRef = useRef<ParticleField | null>(null);

  // Create the field once, after the canvas has its fixed inset-0 dimensions.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const field = createParticleField(canvas);
    fieldRef.current = field;
    return () => {
      field.destroy();
      fieldRef.current = null;
    };
  }, []);

  // Push mode changes to the live field.
  useEffect(() => {
    fieldRef.current?.setMode(mode);
  }, [mode]);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      className="pointer-events-none fixed inset-0 z-0 block"
    />
  );
}
