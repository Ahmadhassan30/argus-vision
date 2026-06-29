/**
 * Vanilla-WebGL particle field for the debate background (no Three.js).
 *
 * ~400 soft particles drift across a fixed fullscreen canvas behind the UI,
 * tinted in the two agent colours. The field reacts to the debate:
 *   • "debate"    — motion speeds up (eased) when the divergence trigger fires;
 *   • "consensus" — particles slow, drift toward centre, and shift to emerald.
 *
 * Delta-time is capped, devicePixelRatio is honoured, and `prefers-reduced-motion`
 * renders a single calm frame instead of animating. All GL resources and the
 * RAF loop are released by the returned `destroy()`.
 */

export type FieldMode = "idle" | "debate" | "consensus";

export interface ParticleField {
  setMode: (mode: FieldMode) => void;
  destroy: () => void;
}

interface RGBA {
  r: number;
  g: number;
  b: number;
  a: number;
}

const AGENT_A: RGBA = { r: 0.145, g: 0.388, b: 0.922, a: 0.06 }; // #2563EB
const AGENT_B: RGBA = { r: 0.486, g: 0.227, b: 0.929, a: 0.06 }; // #7C3AED
const EMERALD: RGBA = { r: 0.02, g: 0.588, b: 0.412, a: 0.08 }; // #059669

const PARTICLE_COUNT = 400;
const MAX_DT = 1 / 30; // cap large frame gaps (tab switches) to keep motion sane.

const VERT = `
attribute vec2 a_pos;
attribute float a_size;
attribute vec4 a_rgba;
uniform vec2 u_res;
varying vec4 v_rgba;
void main() {
  vec2 clip = (a_pos / u_res) * 2.0 - 1.0;
  gl_Position = vec4(clip.x, -clip.y, 0.0, 1.0);
  gl_PointSize = a_size;
  v_rgba = a_rgba;
}`;

const FRAG = `
precision mediump float;
varying vec4 v_rgba;
void main() {
  float d = length(gl_PointCoord - vec2(0.5));
  if (d > 0.5) discard;
  float a = smoothstep(0.5, 0.0, d);
  gl_FragColor = vec4(v_rgba.rgb, v_rgba.a * a);
}`;

function compile(gl: WebGLRenderingContext, type: number, src: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    return null;
  }
  return shader;
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** A pseudo-random sequence seeded for reproducible layout across reloads. */
function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/**
 * Mount a particle field onto `canvas`. Returns controls, or a no-op handle if
 * WebGL is unavailable (the page still works without the background).
 */
export function createParticleField(canvas: HTMLCanvasElement): ParticleField {
  const gl =
    (canvas.getContext("webgl", { alpha: true, premultipliedAlpha: false }) as
      | WebGLRenderingContext
      | null) ?? null;

  const reduced =
    typeof window !== "undefined" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (!gl) {
    return { setMode: () => {}, destroy: () => {} };
  }

  const program = gl.createProgram();
  const vs = compile(gl, gl.VERTEX_SHADER, VERT);
  const fs = compile(gl, gl.FRAGMENT_SHADER, FRAG);
  if (!program || !vs || !fs) {
    return { setMode: () => {}, destroy: () => {} };
  }
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  gl.useProgram(program);

  const aPos = gl.getAttribLocation(program, "a_pos");
  const aSize = gl.getAttribLocation(program, "a_size");
  const aRgba = gl.getAttribLocation(program, "a_rgba");
  const uRes = gl.getUniformLocation(program, "u_res");

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  // CPU-side particle state.
  const rng = makeRng(0x5eed);
  const px = new Float32Array(PARTICLE_COUNT);
  const py = new Float32Array(PARTICLE_COUNT);
  const vx = new Float32Array(PARTICLE_COUNT);
  const vy = new Float32Array(PARTICLE_COUNT);
  const baseR = new Float32Array(PARTICLE_COUNT);
  const baseG = new Float32Array(PARTICLE_COUNT);
  const baseB = new Float32Array(PARTICLE_COUNT);
  const baseA = new Float32Array(PARTICLE_COUNT);
  const size = new Float32Array(PARTICLE_COUNT);

  // Interleaved render buffer: [x, y, size, r, g, b, a] per particle.
  const STRIDE = 7;
  const buffer = new Float32Array(PARTICLE_COUNT * STRIDE);
  const glBuffer = gl.createBuffer();

  let width = 0;
  let height = 0;
  let dpr = 1;

  function resize(): void {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = canvas.clientWidth || window.innerWidth;
    height = canvas.clientHeight || window.innerHeight;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    gl!.viewport(0, 0, canvas.width, canvas.height);
    gl!.uniform2f(uRes, width, height);
  }

  function seed(): void {
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      px[i] = rng() * width;
      py[i] = rng() * height;
      const speed = 6 + rng() * 14; // px/sec, gentle drift
      const angle = rng() * Math.PI * 2;
      vx[i] = Math.cos(angle) * speed;
      vy[i] = Math.sin(angle) * speed;
      const c = rng() < 0.5 ? AGENT_A : AGENT_B;
      baseR[i] = c.r;
      baseG[i] = c.g;
      baseB[i] = c.b;
      baseA[i] = c.a * (0.6 + rng() * 0.8);
      size[i] = (2 + rng() * 4) * dpr;
    }
  }

  resize();
  seed();

  // Eased mode parameters.
  let flow = 0; // 0 idle → 1 debate (speed multiplier driver)
  let resolve = 0; // 0 idle → 1 consensus (colour + convergence driver)
  let targetFlow = 0;
  let targetResolve = 0;

  function render(): void {
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const off = i * STRIDE;
      buffer[off] = px[i];
      buffer[off + 1] = py[i];
      buffer[off + 2] = size[i] * (1 + flow * 0.25);
      buffer[off + 3] = lerp(baseR[i], EMERALD.r, resolve);
      buffer[off + 4] = lerp(baseG[i], EMERALD.g, resolve);
      buffer[off + 5] = lerp(baseB[i], EMERALD.b, resolve);
      buffer[off + 6] = lerp(baseA[i], EMERALD.a * 1.2, resolve * 0.7) * (1 + flow * 0.4);
    }

    gl!.clearColor(0, 0, 0, 0);
    gl!.clear(gl!.COLOR_BUFFER_BIT);
    gl!.bindBuffer(gl!.ARRAY_BUFFER, glBuffer);
    gl!.bufferData(gl!.ARRAY_BUFFER, buffer, gl!.DYNAMIC_DRAW);
    const bytes = STRIDE * 4;
    gl!.enableVertexAttribArray(aPos);
    gl!.vertexAttribPointer(aPos, 2, gl!.FLOAT, false, bytes, 0);
    gl!.enableVertexAttribArray(aSize);
    gl!.vertexAttribPointer(aSize, 1, gl!.FLOAT, false, bytes, 8);
    gl!.enableVertexAttribArray(aRgba);
    gl!.vertexAttribPointer(aRgba, 4, gl!.FLOAT, false, bytes, 12);
    gl!.drawArrays(gl!.POINTS, 0, PARTICLE_COUNT);
  }

  function step(dt: number): void {
    flow += (targetFlow - flow) * Math.min(1, dt * 3.5);
    resolve += (targetResolve - resolve) * Math.min(1, dt * 2.5);
    const cx = width / 2;
    const cy = height / 2;
    const speedMul = 1 + flow * 1.8;

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      px[i] += vx[i] * speedMul * dt;
      py[i] += vy[i] * speedMul * dt;

      // Gentle convergence toward centre as consensus resolves.
      if (resolve > 0.001) {
        px[i] += (cx - px[i]) * resolve * dt * 0.25;
        py[i] += (cy - py[i]) * resolve * dt * 0.25;
      }

      // Wrap around edges with a small margin.
      const m = 8;
      if (px[i] < -m) px[i] = width + m;
      else if (px[i] > width + m) px[i] = -m;
      if (py[i] < -m) py[i] = height + m;
      else if (py[i] > height + m) py[i] = -m;
    }
  }

  let raf = 0;
  let last = 0;
  let running = true;

  function loop(now: number): void {
    if (!running) return;
    const dt = last === 0 ? 0.016 : Math.min((now - last) / 1000, MAX_DT);
    last = now;
    step(dt);
    render();
    raf = requestAnimationFrame(loop);
  }

  const onResize = (): void => resize();
  window.addEventListener("resize", onResize);

  if (reduced) {
    // One calm static frame; no animation loop.
    render();
  } else {
    raf = requestAnimationFrame(loop);
  }

  return {
    setMode(mode: FieldMode) {
      if (mode === "idle") {
        targetFlow = 0;
        targetResolve = 0;
      } else if (mode === "debate") {
        targetFlow = 1;
        targetResolve = 0;
      } else {
        targetFlow = 0.25;
        targetResolve = 1;
      }
      if (reduced) render();
    },
    destroy() {
      running = false;
      if (raf) cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      gl.deleteBuffer(glBuffer);
      gl.deleteProgram(program);
      gl.deleteShader(vs);
      gl.deleteShader(fs);
    },
  };
}
