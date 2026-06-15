"use client";

import { useEffect, useRef, useState } from "react";
import {
  Mic,
  Square,
  Play,
  RefreshCw,
  Eye,
  Loader,
  Volume2,
  Activity,
  Compass,
  Crosshair,
  AlertTriangle,
  Radio,
  Terminal,
  ArrowUp,
  ArrowLeft,
  ArrowRight,
  LucideIcon,
} from "lucide-react";
import { useCamera } from "@/hooks/useCamera";
import { useVoice } from "@/hooks/useVoice";
import { motion, AnimatePresence } from "framer-motion";

type AppMode = "voice" | "scene";

interface TrackerTarget {
  id: number;
  name: string;
  distance: string;
  conf: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
}

export default function CameraPage() {
  const [mode, setMode] = useState<AppMode>("voice");
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const [fps, setFps] = useState(60);
  const [simulatedLatency, setSimulatedLatency] = useState(84);
  const frameIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Simulated target tracking coordinates that drift slightly for realistic AR HUD feel
  const [targets, setTargets] = useState<TrackerTarget[]>([
    { id: 1, name: "PERSON", distance: "2.3m", conf: 98, x: 25, y: 35, vx: 0.04, vy: -0.03 },
    { id: 2, name: "DOORWAY", distance: "4.5m", conf: 92, x: 65, y: 25, vx: -0.02, vy: 0.03 },
    { id: 3, name: "CHAIR", distance: "1.2m", conf: 89, x: 45, y: 65, vx: 0.03, vy: 0.02 },
  ]);

  const {
    videoRef,
    isActive: cameraActive,
    error: cameraError,
    startCamera,
    stopCamera,
    captureCurrentFrame,
  } = useCamera({
    onError: (e) => console.error(e),
  });

  const {
    status,
    transcript,
    latestResponse,
    isConnected,
    startListening,
    stopListening,
    sendFrame,
    clearTranscript,
  } = useVoice({
    mode: mode,
    onError: (msg) => {
      console.error("CameraPage useVoice onError:", msg);
      setVoiceError(msg);
      setTimeout(() => setVoiceError(null), 5000);
    }
  });

  // ── Auto-start camera on mount ─────────────────────────────────────────────
  useEffect(() => {
    startCamera();
  }, [startCamera]);

  // ── Scene mode: send frames every 2.5s ─────────────────────────────────────
  useEffect(() => {
    if (mode === "scene" && cameraActive) {
      const id = setInterval(() => {
        const frame = captureCurrentFrame(0.7);
        if (frame) sendFrame(frame);
      }, 2500);
      frameIntervalRef.current = id;
      return () => clearInterval(id);
    } else {
      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
    }
  }, [mode, cameraActive, captureCurrentFrame, sendFrame]);

  // Capture frame once when user starts speaking (status becomes "listening")
  useEffect(() => {
    if (status === "listening" && mode === "voice") {
      if (cameraActive) {
        console.log("CameraPage: Voice mode listening started. Capturing image...");
        const frame = captureCurrentFrame(0.7);
        if (frame) {
          sendFrame(frame);
        } else {
          sendFrame("");
        }
      } else {
        console.log("CameraPage: Camera not active. Clearing frame on backend...");
        sendFrame("");
      }
    }
  }, [status, cameraActive, mode, captureCurrentFrame, sendFrame]);

  // ── Keyboard shortcut: Space = push-to-talk ───────────────────────────────
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === "Space" && e.target === document.body) {
        e.preventDefault();
        if (
          status === "idle" ||
          status === "error" ||
          status === "speaking" ||
          status === "processing"
        ) {
          startListening();
        }
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === "Space" && status === "listening") {
        e.preventDefault();
        stopListening();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [status, startListening, stopListening]);

  const isHoldingRef = useRef(false);
  const startPressTimeRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Clean up timers on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, []);

  const handlePointerDown = (e: React.MouseEvent | React.TouchEvent) => {
    if ("button" in e && e.button !== 0) return;
    e.preventDefault();

    isHoldingRef.current = false;
    startPressTimeRef.current = Date.now();

    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }

    timerRef.current = setTimeout(() => {
      isHoldingRef.current = true;
      if (
        status === "idle" ||
        status === "error" ||
        status === "speaking" ||
        status === "processing"
      ) {
        startListening();
      }
    }, 250);
  };

  const handlePointerUp = (e: React.MouseEvent | React.TouchEvent) => {
    if ("button" in e && e.button !== 0) return;
    e.preventDefault();

    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }

    if (isHoldingRef.current) {
      isHoldingRef.current = false;
      if (status === "listening") {
        stopListening();
      }
    } else {
      if (status === "listening") {
        stopListening();
      } else {
        startListening();
      }
    }
  };

  const handlePointerLeave = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }

    if (isHoldingRef.current) {
      isHoldingRef.current = false;
      if (status === "listening") {
        stopListening();
      }
    }
  };

  const handleVoiceButtonClick = (e: React.MouseEvent) => {
    if (e.detail === 0) {
      if (status === "listening") {
        stopListening();
      } else {
        startListening();
      }
    }
  };

  // Calculate real-time FPS
  useEffect(() => {
    let lastTime = performance.now();
    let frames = 0;
    let frameId: number;
    const tick = () => {
      frames++;
      const now = performance.now();
      if (now - lastTime >= 1000) {
        setFps(Math.round((frames * 1000) / (now - lastTime)));
        frames = 0;
        lastTime = now;
      }
      frameId = requestAnimationFrame(tick);
    };
    frameId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameId);
  }, []);

  // Fluctuating simulated latency for HUD realism
  useEffect(() => {
    const interval = setInterval(() => {
      setSimulatedLatency((prev) => {
        const drift = Math.random() > 0.5 ? 2 : -2;
        const target = status === "processing" ? 115 : 72;
        return Math.max(45, Math.min(180, Math.round(prev + (target - prev) * 0.2 + drift)));
      });
    }, 800);
    return () => clearInterval(interval);
  }, [status]);

  // Target drift animation loop
  useEffect(() => {
    let frameId: number;
    let active = true;
    const updatePositions = () => {
      if (!active || !cameraActive) return;
      setTargets((prev) =>
        prev.map((t) => {
          const nx = t.x + t.vx;
          const ny = t.y + t.vy;
          let nvx = t.vx;
          let nvy = t.vy;
          if (nx < 15 || nx > 85) nvx = -nvx;
          if (ny < 15 || ny > 85) nvy = -nvy;
          return {
            ...t,
            x: Math.max(15, Math.min(85, nx)),
            y: Math.max(15, Math.min(85, ny)),
            vx: nvx,
            vy: nvy,
          };
        })
      );
      frameId = requestAnimationFrame(updatePositions);
    };
    frameId = requestAnimationFrame(updatePositions);
    return () => {
      active = false;
      cancelAnimationFrame(frameId);
    };
  }, [cameraActive]);

  // Dynamically map detected nouns from VLM response to target indicators
  useEffect(() => {
    if (latestResponse) {
      const words = latestResponse.toLowerCase().split(/\W+/);
      const candidates = [
        "person", "door", "chair", "table", "laptop", "phone",
        "bottle", "backpack", "glass", "staircase", "obstacle",
        "wall", "window", "hazard", "screen", "keyboard", "mouse"
      ];
      const detected = candidates.filter(c => words.includes(c));
      if (detected.length > 0) {
        setTargets((prev) =>
          prev.map((t, idx) => {
            const label = detected[idx % detected.length];
            const dist = `${(0.9 + Math.random() * 2.8).toFixed(1)}m`;
            const confidence = Math.floor(91 + Math.random() * 8);
            return {
              ...t,
              name: label.toUpperCase(),
              distance: dist,
              conf: confidence,
            };
          })
        );
      }
    }
  }, [latestResponse]);

  // Safe path directions extracted from latest description or status
  const getSafePathDirection = (): { text: string; icon: LucideIcon } | null => {
    if (!latestResponse) return null;
    const desc = latestResponse.toLowerCase();
    if (desc.includes("turn left") || desc.includes("left side clear")) {
      return { text: "Turn Left", icon: ArrowLeft };
    }
    if (desc.includes("turn right") || desc.includes("right side clear")) {
      return { text: "Turn Right", icon: ArrowRight };
    }
    if (desc.includes("safe area") || desc.includes("door ahead") || desc.includes("path ahead")) {
      return { text: "Move Forward", icon: ArrowUp };
    }
    return { text: "Move Forward", icon: ArrowUp };
  };

  const pathDirection = getSafePathDirection();

  return (
    <div id="camera-hud-container" className="fixed inset-0 w-screen h-screen overflow-hidden bg-black select-none">
      
      {/* ── 1. Full-Screen Live Video Stream ─────────────────────────────────── */}
      <div className="absolute inset-0 w-full h-full z-0 bg-black">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover transition-opacity duration-700 ${
            cameraActive ? "opacity-100" : "opacity-30"
          }`}
          aria-label="Head-up display camera stream"
        />

        {/* Video placeholder when camera is not running */}
        {!cameraActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 text-cyan-500/80">
            <Compass className="w-16 h-16 animate-pulse text-cyan-400" />
            <p className="text-sm tracking-widest uppercase font-mono text-cyan-400/70">
              {cameraError ?? "Camera HUD Standby"}
            </p>
            <button
              onClick={() => startCamera()}
              className="px-6 py-2.5 bg-cyan-950/40 border border-cyan-500/50 hover:bg-cyan-500/20 text-cyan-400 hover:text-white rounded-sm font-mono text-xs uppercase tracking-wider transition-all duration-300 backdrop-blur-md"
              style={{ boxShadow: "0 0 15px rgba(0, 212, 255, 0.15)" }}
            >
              Initialize Camera Feed
            </button>
          </div>
        )}
      </div>

      {/* ── 2. HUD Aesthetics overlays (Vignette, Grid, Scanlines) ─────────────── */}
      <div className="absolute inset-0 pointer-events-none z-10 hud-grid-overlay opacity-80" />
      <div className="absolute inset-0 pointer-events-none z-10 hud-scanlines opacity-15" />
      <div className="absolute inset-0 pointer-events-none z-10 bg-[radial-gradient(circle_at_center,transparent_40%,rgba(0,0,0,0.6)_100%)]" />

      {/* ── 3. Target lock markers / SVG tracking HUD ────────────────────────── */}
      {cameraActive && (
        <svg className="absolute inset-0 w-full h-full z-20 pointer-events-none">
          {targets.map((t) => (
            <g key={t.id} style={{ transition: "transform 0.05s linear" }}>
              {/* Corner Brackets */}
              <path
                d={`M ${t.x - 3}% ${t.y - 3}% L ${t.x - 1.5}% ${t.y - 3}% M ${t.x - 3}% ${t.y - 3}% L ${t.x - 3}% ${t.y - 1.5}%`}
                stroke="#00d4ff"
                strokeWidth="2"
                fill="none"
              />
              <path
                d={`M ${t.x + 3}% ${t.y - 3}% L ${t.x + 1.5}% ${t.y - 3}% M ${t.x + 3}% ${t.y - 3}% L ${t.x + 3}% ${t.y - 1.5}%`}
                stroke="#00d4ff"
                strokeWidth="2"
                fill="none"
              />
              <path
                d={`M ${t.x - 3}% ${t.y + 3}% L ${t.x - 1.5}% ${t.y + 3}% M ${t.x - 3}% ${t.y + 3}% L ${t.x - 3}% ${t.y + 1.5}%`}
                stroke="#00d4ff"
                strokeWidth="2"
                fill="none"
              />
              <path
                d={`M ${t.x + 3}% ${t.y + 3}% L ${t.x + 1.5}% ${t.y + 3}% M ${t.x + 3}% ${t.y + 3}% L ${t.x + 3}% ${t.y + 1.5}%`}
                stroke="#00d4ff"
                strokeWidth="2"
                fill="none"
              />

              {/* Reticle Dot */}
              <circle cx={`${t.x}%`} cy={`${t.y}%`} r="2" fill="#00d4ff" className="hud-pulse-lock" />

              {/* Text Indicators */}
              <foreignObject
                x={`${t.x + 4}%`}
                y={`${t.y - 3}%`}
                width="140"
                height="60"
                className="overflow-visible"
              >
                <div className="flex flex-col text-[10px] font-mono text-cyan-400 bg-black/60 px-1.5 py-1 rounded border border-cyan-500/30 w-fit backdrop-blur-sm">
                  <span className="font-bold tracking-wider">{t.name}</span>
                  <div className="flex gap-2 text-[8px] opacity-80 mt-0.5">
                    <span>{t.distance}</span>
                    <span>{t.conf}%</span>
                  </div>
                </div>
              </foreignObject>
            </g>
          ))}
        </svg>
      )}

      {/* ── 4. Glowing navigation path overlay ───────────────────────────────── */}
      {cameraActive && (
        <svg className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] z-10 pointer-events-none opacity-45">
          {/* Safe Path perspective lanes */}
          <path
            d="M 100 300 L 250 120 L 350 120 L 500 300 Z"
            fill="url(#hud-path-grad)"
            className="hud-path-flow"
          />
          {/* Grid lines */}
          <line x1="250" y1="120" x2="350" y2="120" stroke="#00d4ff" strokeWidth="1.5" />
          <line x1="220" y1="160" x2="380" y2="160" stroke="#00d4ff" strokeWidth="1" strokeDasharray="4 4" />
          <line x1="180" y1="210" x2="420" y2="210" stroke="#00d4ff" strokeWidth="1" strokeDasharray="4 4" />
          <line x1="130" y1="260" x2="470" y2="260" stroke="#00d4ff" strokeWidth="1" strokeDasharray="4 4" />

          {/* Perspective lanes */}
          <line x1="250" y1="120" x2="100" y2="300" stroke="#00d4ff" strokeWidth="2" />
          <line x1="300" y1="120" x2="300" y2="300" stroke="#00d4ff" strokeWidth="0.5" strokeDasharray="10 5" />
          <line x1="350" y1="120" x2="500" y2="300" stroke="#00d4ff" strokeWidth="2" />

          <defs>
            <linearGradient id="hud-path-grad" x1="0%" y1="100%" x2="0%" y2="0%">
              <stop offset="0%" stopColor="rgba(0,212,255,0.25)" />
              <stop offset="100%" stopColor="rgba(0,212,255,0.0)" />
            </linearGradient>
          </defs>
        </svg>
      )}

      {/* ── 5. Navigation path overlay text ─────────────────────────────────── */}
      {cameraActive && pathDirection && (
        <div className="absolute top-[32%] left-1/2 -translate-x-1/2 z-30 pointer-events-none flex flex-col items-center text-emerald-400 hud-glow-green">
          <pathDirection.icon className="w-8 h-8 animate-bounce mb-1" />
          <span className="font-mono text-xs uppercase tracking-widest font-bold bg-black/60 px-3 py-1 border border-emerald-500/30 rounded backdrop-blur-sm">
            {pathDirection.text}
          </span>
        </div>
      )}

      {/* ── 6. UI Overlays (HUD System Panels) ───────────────────────────────── */}
      <div className="absolute inset-0 w-full h-full z-30 pointer-events-none p-4 flex flex-col justify-between">
        
        {/* TOP SECTION */}
        <div className="flex justify-between items-start">
          
          {/* Top Left: System Status */}
          <div className="ar-hud-panel p-4 w-52 flex flex-col gap-1.5 pointer-events-auto">
            <div className="flex items-center gap-2 border-b border-cyan-500/20 pb-1.5 mb-1.5">
              <Activity className="w-4 h-4 text-cyan-400 animate-pulse" />
              <span className="font-mono text-xs text-white/50 tracking-wider font-bold">SYSTEM TELEMETRY</span>
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-white/40">SYS STAT:</span>
              <span className="text-cyan-400 font-bold tracking-wider">ONLINE</span>
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-white/40">CAMERA:</span>
              <span className={cameraActive ? "text-emerald-400 font-semibold" : "text-white/30"}>
                {cameraActive ? "ACTIVE" : "STANDBY"}
              </span>
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-white/40">AUDIO LNK:</span>
              <span className={isConnected ? "text-emerald-400 font-semibold" : "text-rose-500 font-semibold"}>
                {isConnected ? "ACTIVE" : "ERR"}
              </span>
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-white/40">AI SYSTEM:</span>
              <span className="text-cyan-400">READY</span>
            </div>
          </div>

          {/* Top Right: Mode selector & toggle */}
          <div className="ar-hud-panel p-4 w-60 flex flex-col gap-2 pointer-events-auto items-end">
            <div className="flex items-center gap-2 border-b border-cyan-500/20 pb-1.5 mb-1.5 w-full justify-end">
              <span className="font-mono text-xs text-white/50 tracking-wider font-bold">TARGET MODE</span>
              <Radio className="w-4 h-4 text-cyan-400 animate-pulse" />
            </div>

            {/* Mode selection buttons */}
            <div className="flex gap-2 w-full">
              <button
                onClick={() => setMode("voice")}
                className="flex-1 px-3 py-1.5 border font-mono text-[10px] uppercase tracking-wider transition-all duration-300 rounded"
                style={{
                  background: mode === "voice" ? "rgba(0,212,255,0.15)" : "transparent",
                  borderColor: mode === "voice" ? "#00d4ff" : "rgba(255,255,255,0.08)",
                  color: mode === "voice" ? "#00d4ff" : "#64748b",
                  textShadow: mode === "voice" ? "0 0 8px rgba(0, 212, 255, 0.4)" : "none",
                }}
              >
                Voice Q&A
              </button>
              <button
                onClick={() => setMode("scene")}
                className="flex-1 px-3 py-1.5 border font-mono text-[10px] uppercase tracking-wider transition-all duration-300 rounded"
                style={{
                  background: mode === "scene" ? "rgba(0,212,255,0.15)" : "transparent",
                  borderColor: mode === "scene" ? "#00d4ff" : "rgba(255,255,255,0.08)",
                  color: mode === "scene" ? "#00d4ff" : "#64748b",
                  textShadow: mode === "scene" ? "0 0 8px rgba(0, 212, 255, 0.4)" : "none",
                }}
              >
                Scene Mode
              </button>
            </div>
            
            <div className="text-[9px] font-mono text-white/30 text-right mt-1">
              MODEL: nvidia/llama-3.1-nemotron-nano-vl
            </div>
          </div>
        </div>

        {/* MIDDLE SECTION: Radial scanner radar */}
        <div className="absolute top-[42%] left-[45%] -translate-x-1/2 -translate-y-1/2 pointer-events-none flex items-center justify-center">
          <AnimatePresence>
            {status === "processing" && (
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 0.75 }}
                exit={{ scale: 1.2, opacity: 0 }}
                transition={{ duration: 0.4 }}
                className="relative w-48 h-48 rounded-full border border-cyan-500/20 flex items-center justify-center"
              >
                {/* Outer scanning circle */}
                <div className="absolute inset-2 border-2 border-dashed border-cyan-400/40 rounded-full animate-spin [animation-duration:12s]" />
                
                {/* Rotating scanner sweep line */}
                <div className="absolute inset-0 hud-radar-spin">
                  <div
                    className="w-1/2 h-full bg-gradient-to-r from-cyan-500/40 to-transparent"
                    style={{
                      clipPath: "polygon(50% 50%, 100% 0, 100% 30%, 50% 50%)",
                      transformOrigin: "center center",
                    }}
                  />
                </div>

                {/* Target Crosshair */}
                <Crosshair className="w-8 h-8 text-cyan-400 animate-pulse" />
                <span className="absolute -bottom-6 font-mono text-[9px] text-cyan-400 tracking-widest font-bold uppercase animate-pulse">
                  SCANNING ENVIRONMENT...
                </span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* BOTTOM SECTION */}
        <div className="flex justify-between items-end gap-6 z-30">
          
          {/* Bottom Left: Metrics telemetry */}
          <div className="ar-hud-panel p-4 w-56 flex flex-col gap-1.5 pointer-events-auto">
            <div className="flex items-center gap-2 border-b border-cyan-500/20 pb-1.5 mb-1.5">
              <Terminal className="w-4 h-4 text-cyan-400" />
              <span className="font-mono text-xs text-white/50 tracking-wider font-bold">DIAGNOSTICS</span>
            </div>
            
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-white/40">SYSTEM FPS:</span>
              <span className="text-cyan-400 hud-mono">{fps}</span>
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-white/40">VLM LATENCY:</span>
              <span className="text-cyan-400 hud-mono">{simulatedLatency}ms</span>
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-white/40">ACTIVE LOCKS:</span>
              <span className="text-cyan-400 hud-mono">{cameraActive ? targets.length : 0}</span>
            </div>
            
            {/* Tiny circular radar scan widget */}
            <div className="flex justify-between items-center mt-2 pt-2 border-t border-cyan-500/10">
              <span className="text-[9px] font-mono text-white/30 uppercase tracking-widest">Radar Overlay</span>
              <div className="w-7 h-7 rounded-full border border-cyan-400/30 relative overflow-hidden flex items-center justify-center">
                <div className="absolute inset-0 bg-cyan-950/20" />
                <div className="w-full h-full absolute hud-radar-spin border-r border-cyan-400/50" style={{ transformOrigin: "center center" }} />
                <div className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
              </div>
            </div>
          </div>

          {/* Bottom Center: Voice assistant controller */}
          <div className="flex-1 max-w-xl flex flex-col items-center gap-3">
            
            {/* Visualizer and text outputs */}
            <div className="w-full flex flex-col items-center bg-black/60 border border-cyan-500/20 rounded px-4 py-3 min-h-24 backdrop-blur-md relative">
              <div className="absolute top-1.5 right-2 flex gap-1 items-center font-mono text-[9px] text-white/30">
                <div className={`w-1.5 h-1.5 rounded-full ${isConnected ? "bg-emerald-400" : "bg-rose-500 animate-pulse"}`} />
                {isConnected ? "WS CONN" : "NO CONN"}
              </div>

              {/* Status Header */}
              <div className="font-mono text-[10px] text-cyan-400/70 tracking-widest uppercase mb-1 font-bold">
                {status === "listening" ? (
                  <span className="text-cyan-400 animate-pulse">🎤 REC RECEIVER ACTIVE</span>
                ) : status === "processing" ? (
                  <span className="text-amber-400 animate-pulse">⚡ COMPILING AI QUERY</span>
                ) : status === "speaking" ? (
                  <span className="text-emerald-400 animate-pulse">🤖 TRANSMITTING RESPONSE</span>
                ) : (
                  <span className="text-white/40">📡 SYSTEM RECEIVER STANDBY</span>
                )}
              </div>

              {/* Voice transcript block */}
              <div className="w-full text-center text-white/95 min-h-6 flex items-center justify-center font-medium">
                {status === "listening" && transcript.length > 0 && (
                  <span className="text-base text-cyan-200 tracking-wide">
                    &quot;{transcript[transcript.length - 1]?.text}&quot;
                  </span>
                )}
                {status === "listening" && transcript.length === 0 && (
                  <span className="text-sm italic text-cyan-400/40 tracking-wider">Listening to speech...</span>
                )}
                {status === "speaking" && latestResponse && (
                  <span className="text-base text-emerald-300 font-semibold tracking-wide">
                    &quot;{latestResponse}&quot;
                  </span>
                )}
                {status === "processing" && (
                  <span className="text-xs text-amber-300 font-mono tracking-widest uppercase animate-pulse">
                    Querying VLM nemotron model...
                  </span>
                )}
                {status === "idle" && (
                  <span className="text-sm text-white/40 italic">
                    Press mic or hold Space to trigger visual query
                  </span>
                )}
              </div>

              {/* Pulse Audio Waveform (neon colors) */}
              <div className="flex items-center gap-1.5 h-6 mt-2 justify-center w-full">
                {status === "listening" && (
                  <>
                    <div className="wave-bar bg-cyan-400" style={{ animationDelay: "0.1s" }} />
                    <div className="wave-bar bg-cyan-400" style={{ animationDelay: "0.3s" }} />
                    <div className="wave-bar bg-cyan-400" style={{ animationDelay: "0.5s" }} />
                    <div className="wave-bar bg-cyan-400" style={{ animationDelay: "0.2s" }} />
                    <div className="wave-bar bg-cyan-400" style={{ animationDelay: "0.4s" }} />
                    <div className="wave-bar bg-cyan-400" style={{ animationDelay: "0.6s" }} />
                  </>
                )}
                {status === "speaking" && (
                  <>
                    <div className="wave-bar bg-emerald-400" style={{ animationDelay: "0.2s", height: "14px" }} />
                    <div className="wave-bar bg-emerald-400" style={{ animationDelay: "0.4s", height: "18px" }} />
                    <div className="wave-bar bg-emerald-400" style={{ animationDelay: "0.1s", height: "12px" }} />
                    <div className="wave-bar bg-emerald-400" style={{ animationDelay: "0.3s", height: "16px" }} />
                    <div className="wave-bar bg-emerald-400" style={{ animationDelay: "0.5s", height: "10px" }} />
                  </>
                )}
                {status === "idle" && (
                  <div className="h-0.5 w-16 bg-cyan-500/30 rounded" />
                )}
              </div>
            </div>

            {/* Central Activator Mic Button */}
            <div className="flex items-center gap-4 pointer-events-auto">
              {/* Back to main controls */}
              <button
                onClick={() => {
                  if (cameraActive) stopCamera();
                  else startCamera();
                }}
                className="p-3 bg-black/60 border border-cyan-500/30 hover:border-cyan-400 rounded-full text-cyan-400 transition-all backdrop-blur-md"
                aria-label={cameraActive ? "Stop Camera feed" : "Start Camera feed"}
              >
                {cameraActive ? <Square size={16} /> : <Play size={16} />}
              </button>

              <button
                id="voice-btn"
                onMouseDown={handlePointerDown}
                onMouseUp={handlePointerUp}
                onMouseLeave={handlePointerLeave}
                onTouchStart={handlePointerDown}
                onTouchEnd={handlePointerUp}
                onTouchCancel={handlePointerLeave}
                onClick={handleVoiceButtonClick}
                className={`w-20 h-20 rounded-full border flex items-center justify-center transition-all duration-300 ${
                  status === "listening"
                    ? "bg-rose-950/40 border-rose-500 text-rose-400 shadow-[0_0_30px_rgba(244,63,94,0.4)] animate-pulse"
                    : status === "processing"
                    ? "bg-amber-950/40 border-amber-500 text-amber-400 shadow-[0_0_30px_rgba(245,158,11,0.4)] animate-pulse"
                    : "bg-cyan-950/40 border-cyan-500 text-cyan-400 hover:border-cyan-300 shadow-[0_0_20px_rgba(0,212,255,0.2)] hover:shadow-[0_0_35px_rgba(0,212,255,0.4)]"
                } backdrop-blur-md`}
                aria-label={status === "listening" ? "Transmitting audio..." : "Trigger Voice Query"}
                disabled={!isConnected && status !== "connecting"}
              >
                {status === "processing" ? (
                  <Loader size={28} className="animate-spin text-amber-400" />
                ) : status === "speaking" ? (
                  <Volume2 size={28} className="text-emerald-400 animate-pulse" />
                ) : (
                  <Mic size={28} className={status === "listening" ? "text-rose-400" : "text-cyan-400"} />
                )}
              </button>

              {/* Clear History Button */}
              <button
                onClick={clearTranscript}
                className="p-3 bg-black/60 border border-cyan-500/30 hover:border-cyan-400 rounded-full text-cyan-400 transition-all backdrop-blur-md"
                aria-label="Clear chat telemetry"
              >
                <RefreshCw size={16} />
              </button>
            </div>
          </div>

          {/* Bottom Right: Real-time Scene Analysis Glaspanel */}
          <div className="ar-hud-panel p-4 w-64 h-48 flex flex-col pointer-events-auto overflow-hidden">
            <div className="flex items-center gap-2 border-b border-cyan-500/20 pb-1.5 mb-1.5">
              <Eye className="w-4 h-4 text-cyan-400" />
              <span className="font-mono text-xs text-white/50 tracking-wider font-bold">AI ANALYSIS LOG</span>
            </div>

            <div className="flex-1 overflow-y-auto pr-1 text-[11px] font-mono text-cyan-300/80 space-y-1.5">
              {latestResponse ? (
                <div className="animate-fade-in leading-relaxed">
                  {latestResponse.split(/[.!?]\s+/).filter(Boolean).map((sentence, idx) => (
                    <div key={idx} className="flex gap-1.5 items-start">
                      <span className="text-cyan-400">⚡</span>
                      <span>{sentence.trim()}.</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="space-y-1.5 opacity-60">
                  <div className="flex gap-1.5 items-center">
                    <span className="text-cyan-400/40">⚡</span>
                    <span className="text-white/40">Scene telemetry standby...</span>
                  </div>
                  <div className="flex gap-1.5 items-center">
                    <span className="text-cyan-400/40">⚡</span>
                    <span className="text-white/40">Grid calibration nominal.</span>
                  </div>
                  <div className="flex gap-1.5 items-center text-cyan-400 animate-pulse">
                    <span>•</span>
                    <span>Ready for voice query start.</span>
                  </div>
                </div>
              )}
            </div>
          </div>
          
        </div>
      </div>
      
      {/* Voice connection or system error alerts */}
      {voiceError && (
        <div
          className="fixed bottom-36 left-4 z-40 px-4 py-2 bg-rose-950/80 border border-rose-500 text-rose-300 text-xs font-mono rounded backdrop-blur-md flex items-center gap-2"
          role="alert"
          style={{ boxShadow: "0 0 20px rgba(244, 63, 94, 0.3)" }}
        >
          <AlertTriangle className="w-4 h-4 animate-bounce" />
          <span>ALERT: {voiceError}</span>
        </div>
      )}
    </div>
  );
}
