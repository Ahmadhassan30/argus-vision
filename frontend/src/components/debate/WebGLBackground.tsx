"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/examples/jsm/postprocessing/UnrealBloomPass.js";
import GUI from "lil-gui";

interface WebGLBackgroundProps {
  /** Drives the field's behaviour: "idle" | "debate" | "consensus". */
  mode: "idle" | "debate" | "consensus";
}

export default function WebGLBackground({ mode }: WebGLBackgroundProps): React.JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // References to allow dynamic parameters updates from the parent prop changes
  const paramsRef = useRef({
    colorBg: "#080808",
    colorLine: "#000075", // Deep clinical blue strands
    colorSignal: "#f50529", // Vibrant red signal color
    useColor2: false,
    colorSignal2: "#ff0055",
    useColor3: false,
    colorSignal3: "#ffcc00",
    lineCount: 80,
    globalRotation: 0,
    positionX: 0.0, // Centered horizontally
    positionY: 13.8, // Aligned to pass precisely through ears/temples of head logo
    spreadHeight: 45.01,
    spreadDepth: 0,
    curveLength: 85, // Symmetrical funnel size left
    straightLength: 84.98, // Symmetrical funnel size right
    curvePower: 0.8265,
    waveSpeed: 3.6,
    waveHeight: 0.1,
    lineOpacity: 0.535,
    signalCount: 136,
    speedGlobal: 0.693,
    trailLength: 3,
    bloomStrength: 3.0,
    bloomRadius: 0.5,
  });

  // Track the active mode inside parameters ref
  useEffect(() => {
    const params = paramsRef.current;
    if (mode === "idle") {
      params.colorSignal = "#f50529";
      params.useColor2 = false;
      params.useColor3 = false;
      params.speedGlobal = 0.693;
      params.waveSpeed = 3.6;
      params.waveHeight = 0.1;
      params.bloomStrength = 3.0;
    } else if (mode === "debate") {
      // Speed up and add violent multi-color signals during live arguments
      params.colorSignal = "#f50529";
      params.useColor2 = true;
      params.useColor3 = true;
      params.speedGlobal = 1.25;
      params.waveSpeed = 4.2;
      params.waveHeight = 0.28;
      params.bloomStrength = 4.0;
    } else if (mode === "consensus") {
      // Steady consensus emerald glow
      params.colorSignal = "#10b981";
      params.useColor2 = false;
      params.useColor3 = false;
      params.speedGlobal = 0.15;
      params.waveSpeed = 1.0;
      params.waveHeight = 0.05;
      params.bloomStrength = 5.0; // High bloom glow for the resolution
    }
  }, [mode]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const params = paramsRef.current;
    const segmentCount = 150;

    // --- Scene Setup ---
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(params.colorBg);
    scene.fog = new THREE.FogExp2(params.colorBg, 0.002);

    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 1000);
    camera.position.set(0, 0, 90);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const contentGroup = new THREE.Group();
    contentGroup.position.set(params.positionX, params.positionY, 0);
    scene.add(contentGroup);

    // --- Post-Processing (Bloom) ---
    const renderScene = new RenderPass(scene, camera);
    const bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      params.bloomStrength,
      params.bloomRadius,
      0.85
    );
    bloomPass.threshold = 0;

    const composer = new EffectComposer(renderer);
    composer.addPass(renderScene);
    composer.addPass(bloomPass);

    // --- Math & Path calculation ---
    function getPathPoint(t: number, lineIndex: number, time: number) {
      // Span symmetrically from -curveLength to +straightLength (which are equal: e.g. -85 to +85)
      const totalLen = params.curveLength + params.straightLength;
      const currentX = -params.curveLength + t * totalLen;

      let y = 0;
      let z = 0;
      const spreadFactor = (lineIndex / params.lineCount - 0.5) * 2;

      // Define where the center straight consensus beam lies (e.g. middle 30% of total length)
      const convergeLimit = params.curveLength * 0.32; // e.g. 27 units
      const outerDist = Math.abs(currentX);

      if (outerDist > convergeLimit) {
        // We are in the left or right funnel region
        const activeRange = params.curveLength - convergeLimit;
        const ratio = (outerDist - convergeLimit) / activeRange;
        
        // Symmetrical smooth ease-in curve
        let shapeFactor = (Math.cos((1 - ratio) * Math.PI) + 1) / 2;
        shapeFactor = Math.pow(shapeFactor, params.curvePower);

        y = spreadFactor * params.spreadHeight * shapeFactor;
        z = spreadFactor * params.spreadDepth * shapeFactor;

        // Wave active inside the outer funnels
        const wave =
          Math.sin(time * params.waveSpeed + currentX * 0.1 + lineIndex) *
          params.waveHeight *
          shapeFactor;
        y += wave;
      }

      return new THREE.Vector3(currentX, y, z);
    }

    // --- Objects ---
    let backgroundLines: THREE.Line[] = [];
    interface SignalData {
      mesh: THREE.Line;
      laneIndex: number;
      speed: number;
      progress: number;
      history: THREE.Vector3[];
      assignedColor: THREE.Color;
    }
    let signals: SignalData[] = [];

    const bgMaterial = new THREE.LineBasicMaterial({
      color: params.colorLine,
      transparent: true,
      opacity: params.lineOpacity,
      depthWrite: false,
    });

    const signalMaterial = new THREE.LineBasicMaterial({
      vertexColors: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: false,
      transparent: true,
    });

    const signalColorObj1 = new THREE.Color(params.colorSignal);
    const signalColorObj2 = new THREE.Color(params.colorSignal2);
    const signalColorObj3 = new THREE.Color(params.colorSignal3);

    function pickSignalColor() {
      signalColorObj1.set(params.colorSignal);
      signalColorObj2.set(params.colorSignal2);
      signalColorObj3.set(params.colorSignal3);

      const choices = [signalColorObj1];
      if (params.useColor2) choices.push(signalColorObj2);
      if (params.useColor3) choices.push(signalColorObj3);
      return choices[Math.floor(Math.random() * choices.length)];
    }

    function rebuildLines() {
      backgroundLines.forEach((l) => {
        contentGroup.remove(l);
        l.geometry.dispose();
      });
      backgroundLines = [];

      for (let i = 0; i < params.lineCount; i++) {
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(segmentCount * 3);
        geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));

        const line = new THREE.Line(geometry, bgMaterial);
        line.userData = { id: i };
        line.renderOrder = 0;
        contentGroup.add(line);
        backgroundLines.push(line);
      }
      rebuildSignals();
    }

    function rebuildSignals() {
      signals.forEach((s) => {
        contentGroup.remove(s.mesh);
        s.mesh.geometry.dispose();
      });
      signals = [];
      for (let i = 0; i < params.signalCount; i++) {
        createSignal();
      }
    }

    function createSignal() {
      const maxTrail = 150;
      const geometry = new THREE.BufferGeometry();
      const positions = new Float32Array(maxTrail * 3);
      const colors = new Float32Array(maxTrail * 3);

      geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

      const mesh = new THREE.Line(geometry, signalMaterial);
      mesh.frustumCulled = false;
      mesh.renderOrder = 1;
      contentGroup.add(mesh);

      signals.push({
        mesh: mesh,
        laneIndex: Math.floor(Math.random() * params.lineCount),
        speed: 0.2 + Math.random() * 0.5,
        progress: Math.random(),
        history: [],
        assignedColor: pickSignalColor(),
      });
    }

    rebuildLines();

    // --- GUI Setup ---
    const gui = new GUI({ title: "Visual Control Arena", autoPlace: true });
    gui.domElement.style.position = "absolute";
    gui.domElement.style.top = "16px";
    gui.domElement.style.right = "16px";
    gui.domElement.style.bottom = "auto";
    gui.close(); // Collapsed by default

    const folderColors = gui.addFolder("Colors");
    folderColors.addColor(params, "colorBg").name("Background").onChange((v: string) => {
      scene.background = new THREE.Color(v);
      scene.fog = new THREE.FogExp2(v, 0.002);
    });
    folderColors.addColor(params, "colorLine").name("Lines").onChange((v: string) => {
      bgMaterial.color.set(v);
    });

    const folderSignalColors = gui.addFolder("Signal Colors");
    folderSignalColors.addColor(params, "colorSignal").name("Main Color").onChange((v: string) => signalColorObj1.set(v));
    folderSignalColors.add(params, "useColor2").name("Use Color 2");
    folderSignalColors.addColor(params, "colorSignal2").name("Color 2").onChange((v: string) => signalColorObj2.set(v));
    folderSignalColors.add(params, "useColor3").name("Use Color 3");
    folderSignalColors.addColor(params, "colorSignal3").name("Color 3").onChange((v: string) => signalColorObj3.set(v));

    const folderGeneral = gui.addFolder("General");
    folderGeneral.add(params, "globalRotation", -180, 180).name("Rotation").onChange((v: number) => {
      contentGroup.rotation.z = THREE.MathUtils.degToRad(v);
    });
    folderGeneral.add(params, "positionX", -200, 200).name("Offset X").onChange((v: number) => {
      contentGroup.position.x = v;
    });
    folderGeneral.add(params, "positionY", -100, 100).name("Offset Y").onChange((v: number) => {
      contentGroup.position.y = v;
    });
    folderGeneral.add(params, "lineCount", 10, 300, 1).name("Lanes").onFinishChange(rebuildLines);

    const folderGeo = gui.addFolder("Geometry");
    folderGeo.add(params, "spreadHeight", 10, 100);
    folderGeo.add(params, "spreadDepth", 0, 50);
    folderGeo.add(params, "curveLength", 20, 150);
    folderGeo.add(params, "straightLength", 20, 200);
    folderGeo.add(params, "curvePower", 0.1, 3.0);

    const folderAnim = gui.addFolder("Line Waves");
    folderAnim.add(params, "waveSpeed", 0, 5);
    folderAnim.add(params, "waveHeight", 0, 5);
    folderAnim.add(params, "lineOpacity", 0, 1).onChange((v: number) => (bgMaterial.opacity = v));

    const folderSignals = gui.addFolder("Signals");
    folderSignals.add(params, "signalCount", 0, 200, 1).name("Count").onFinishChange(rebuildSignals);
    folderSignals.add(params, "speedGlobal", 0, 3).name("Speed");
    folderSignals.add(params, "trailLength", 0, 100, 1).name("Trail Length");

    const folderBloom = gui.addFolder("Bloom Post");
    folderBloom.add(params, "bloomStrength", 0, 8).onChange((v: number) => (bloomPass.strength = v));
    folderBloom.add(params, "bloomRadius", 0, 2).onChange((v: number) => (bloomPass.radius = v));

    // --- Animation Loop ---
    const clock = new THREE.Clock();
    let frameId = 0;

    function animate() {
      frameId = requestAnimationFrame(animate);

      const time = clock.getElapsedTime();

      // Dynamic updates from props
      bloomPass.strength = params.bloomStrength;
      bloomPass.radius = params.bloomRadius;

      // 1. Update Lines
      backgroundLines.forEach((line) => {
        const positions = line.geometry.attributes.position.array as Float32Array;
        const lineId = line.userData.id;
        for (let j = 0; j < segmentCount; j++) {
          const t = j / (segmentCount - 1);
          const vec = getPathPoint(t, lineId, time);
          positions[j * 3] = vec.x;
          positions[j * 3 + 1] = vec.y;
          positions[j * 3 + 2] = vec.z;
        }
        line.geometry.attributes.position.needsUpdate = true;
      });

      // 2. Update Signals
      signals.forEach((sig) => {
        sig.progress += sig.speed * 0.005 * params.speedGlobal;

        if (sig.progress > 1.0) {
          sig.progress = 0;
          sig.laneIndex = Math.floor(Math.random() * params.lineCount);
          sig.history = [];
          sig.assignedColor = pickSignalColor();
        }

        const pos = getPathPoint(sig.progress, sig.laneIndex, time);
        sig.history.push(pos);

        if (sig.history.length > params.trailLength + 1) {
          sig.history.shift();
        }

        const positions = sig.mesh.geometry.attributes.position.array as Float32Array;
        const colors = sig.mesh.geometry.attributes.color.array as Float32Array;

        const drawCount = Math.max(1, params.trailLength);
        const currentLen = sig.history.length;

        for (let i = 0; i < drawCount; i++) {
          let index = currentLen - 1 - i;
          if (index < 0) index = 0;

          const p = sig.history[index] || new THREE.Vector3();

          positions[i * 3] = p.x;
          positions[i * 3 + 1] = p.y;
          positions[i * 3 + 2] = p.z;

          let alpha = 1;
          if (params.trailLength > 0) {
            alpha = Math.max(0, 1 - i / params.trailLength);
          }

          colors[i * 3] = sig.assignedColor.r * alpha;
          colors[i * 3 + 1] = sig.assignedColor.g * alpha;
          colors[i * 3 + 2] = sig.assignedColor.b * alpha;
        }

        sig.mesh.geometry.setDrawRange(0, drawCount);
        sig.mesh.geometry.attributes.position.needsUpdate = true;
        sig.mesh.geometry.attributes.color.needsUpdate = true;
      });

      composer.render();
    }

    animate();

    // Resize Handler
    const handleResize = () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
      composer.setSize(window.innerWidth, window.innerHeight);
    };
    window.addEventListener("resize", handleResize);

    // --- Cleanup ---
    return () => {
      cancelAnimationFrame(frameId);
      window.removeEventListener("resize", handleResize);
      gui.destroy();

      // Dispose resources
      backgroundLines.forEach((l) => l.geometry.dispose());
      signals.forEach((s) => s.mesh.geometry.dispose());
      bgMaterial.dispose();
      signalMaterial.dispose();
      renderer.dispose();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      aria-hidden
      className="pointer-events-none absolute inset-0 z-0 block h-full w-full"
    />
  );
}
