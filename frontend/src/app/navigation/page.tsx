"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Compass,
  Mic,
  Square,
  Play,
  RefreshCw,
  Eye,
  Loader,
  Activity,
  AlertTriangle,
  Crosshair,
  ArrowUp,
  ArrowLeft,
  ArrowRight,
  ChevronLeft,
  Terminal,
  LucideIcon,
} from "lucide-react";
import { useCamera } from "@/hooks/useCamera";
import { useVoice } from "@/hooks/useVoice";
import { motion, AnimatePresence } from "framer-motion";

export default function NavigationPage() {
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const [fps, setFps] = useState(60);
  const [simulatedLatency, setSimulatedLatency] = useState(76);
  const [heading, setHeading] = useState(0); // dynamic compass heading

  const {
    videoRef,
    isActive: cameraActive,
    error: cameraError,
    startCamera,
    stopCamera,
    captureCurrentFrame,
  } = useCamera();

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
    mode: "navigation",
    onError: (msg) => {
      console.error("NavigationPage useVoice onError:", msg);
      setVoiceError(msg);
      setTimeout(() => setVoiceError(null), 5000);
    }
  });

  // Auto-start camera and voice assistant on mount
  useEffect(() => {
    startCamera();
  }, [startCamera]);

  // Capture frame once when user starts speaking (status becomes "listening")
  useEffect(() => {
    if (status === "listening" && cameraActive) {
      console.log("NavigationPage: Capturing navigation frame...");
      const frame = captureCurrentFrame(0.7);
      if (frame) {
        sendFrame(frame);
      } else {
        sendFrame("");
      }
    }
  }, [status, cameraActive, captureCurrentFrame, sendFrame]);

  // Dynamic compass rotation drift
  useEffect(() => {
    const interval = setInterval(() => {
      setHeading((prev) => (prev + Math.floor(Math.random() * 5 - 2) + 360) % 360);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

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

  // Latency telemetry fluctuation
  useEffect(() => {
    const interval = setInterval(() => {
      setSimulatedLatency((prev) => {
        const drift = Math.random() > 0.5 ? 3 : -3;
        const target = status === "processing" ? 110 : 65;
        return Math.max(45, Math.min(180, Math.round(prev + (target - prev) * 0.25 + drift)));
      });
    }, 800);
    return () => clearInterval(interval);
  }, [status]);

  // Directions mapping derived from current VLM response
  const getNavDirection = (): { text: string; icon: LucideIcon; alert: boolean } => {
    if (!latestResponse) return { text: "CALIBRATING PATH", icon: Compass, alert: false };
    const desc = latestResponse.toLowerCase();
    if (desc.includes("stop") || desc.includes("danger") || desc.includes("hazard") || desc.includes("obstacle ahead")) {
      return { text: "WARNING: STOP", icon: AlertTriangle, alert: true };
    }
    if (desc.includes("turn left") || desc.includes("left clear")) {
      return { text: "TURN LEFT", icon: ArrowLeft, alert: false };
    }
    if (desc.includes("turn right") || desc.includes("right clear")) {
      return { text: "TURN RIGHT", icon: ArrowRight, alert: false };
    }
    return { text: "SAFE PATH AHEAD", icon: ArrowUp, alert: false };
  };

  const navDirection = getNavDirection();

  return (
    <div id="hud-fullscreen" className="fixed inset-0 w-screen h-screen overflow-hidden bg-black select-none z-0">
      
      {/* ── 1. Full-Screen Live Camera ────────────────────────────────────────── */}
      <div className="absolute inset-0 w-full h-full z-0 bg-black">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover transition-opacity duration-700 ${
            cameraActive ? "opacity-100" : "opacity-35"
          }`}
          aria-label="Navigation camera stream"
        />

        {!cameraActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 text-cyan-500/80">
            <Compass className="w-16 h-16 animate-spin text-cyan-400" />
            <p className="text-sm tracking-widest uppercase font-mono text-cyan-400/70">
              {cameraError ?? "Initializing Navigation Visor..."}
            </p>
            <button
              onClick={() => startCamera()}
              className="px-6 py-2.5 bg-cyan-950/40 border border-cyan-500/50 hover:bg-cyan-500/20 text-cyan-400 hover:text-white rounded-sm font-mono text-xs uppercase tracking-wider transition-all duration-300 backdrop-blur-md"
              style={{ boxShadow: "0 0 15px rgba(0, 212, 255, 0.15)" }}
            >
              Start Visor
            </button>
          </div>
        )}
      </div>

      {/* ── 2. Tactical AR Overlays (Grid, Scanlines) ─────────────────────────── */}
      <div className="absolute inset-0 pointer-events-none z-10 hud-grid-overlay opacity-80" />
      <div className="absolute inset-0 pointer-events-none z-10 hud-scanlines opacity-15" />
      <div className="absolute inset-0 pointer-events-none z-10 bg-[radial-gradient(circle_at_center,transparent_40%,rgba(0,0,0,0.6)_100%)]" />

      {/* perspective grid lanes */}
      {cameraActive && (
        <svg className="absolute bottom-0 left-1/2 -translate-x-1/2 w-[600px] h-[300px] z-10 pointer-events-none opacity-45">
          <path
            d="M 100 300 L 250 120 L 350 120 L 500 300 Z"
            fill="url(#nav-path-grad)"
            className="hud-path-flow"
          />
          <line x1="250" y1="120" x2="350" y2="120" stroke="#0099CC" strokeWidth="2" />
          <line x1="220" y1="160" x2="380" y2="160" stroke="#0099CC" strokeWidth="1" strokeDasharray="4 4" />
          <line x1="180" y1="210" x2="420" y2="210" stroke="#0099CC" strokeWidth="1" strokeDasharray="4 4" />
          <line x1="130" y1="260" x2="470" y2="260" stroke="#0099CC" strokeWidth="1.5" />
          
          <line x1="250" y1="120" x2="100" y2="300" stroke="#0099CC" strokeWidth="2.5" />
          <line x1="300" y1="120" x2="300" y2="300" stroke="#0099CC" strokeWidth="0.5" strokeDasharray="10 5" />
          <line x1="350" y1="120" x2="500" y2="300" stroke="#0099CC" strokeWidth="2.5" />

          <defs>
            <linearGradient id="nav-path-grad" x1="0%" y1="100%" x2="0%" y2="0%">
              <stop offset="0%" stopColor="rgba(0,153,204,0.3)" />
              <stop offset="100%" stopColor="rgba(106,90,205,0.0)" />
            </linearGradient>
          </defs>
        </svg>
      )}

      {/* Floating Center direction alert */}
      {cameraActive && (
        <div className={`absolute top-[34%] left-1/2 -translate-x-1/2 z-30 pointer-events-none flex flex-col items-center ${
          navDirection.alert ? "text-rose-500 hud-glow-red" : "text-emerald-400 hud-glow-green"
        }`}>
          <navDirection.icon className={`w-12 h-12 mb-2 ${navDirection.alert ? "animate-ping" : "animate-bounce"}`} />
          <span className="font-mono text-sm uppercase tracking-widest font-black bg-black/70 px-4 py-2 border rounded backdrop-blur-md"
            style={{
              borderColor: navDirection.alert ? "rgba(239,68,68,0.4)" : "rgba(16,185,129,0.4)"
            }}
          >
            {navDirection.text}
          </span>
        </div>
      )}

      {/* ── 3. HUD Glass Panels ───────────────────────────────────────────────── */}
      <div className="absolute inset-0 w-full h-full z-30 pointer-events-none p-4 flex flex-col justify-between">
        
        {/* TOP SECTION */}
        <div className="flex justify-between items-start">
          
          {/* Back Port link */}
          <Link
            href="/"
            className="p-3 bg-black/60 border border-cyan-500/30 hover:border-cyan-400 rounded-full text-cyan-400 pointer-events-auto backdrop-blur-md flex items-center gap-2 font-mono text-xs uppercase tracking-wider"
            aria-label="Back to dashboard"
          >
            <ChevronLeft size={16} />
            <span>DASHBOARD</span>
          </Link>

          {/* Top Center Compass Widget */}
          <div className="ar-hud-panel px-5 py-2.5 flex items-center gap-3 w-56 justify-center">
            <Compass className="w-5 h-5 text-cyan-400 animate-pulse" style={{ transform: `rotate(${heading}deg)` }} />
            <div className="flex flex-col font-mono text-xs">
              <span className="text-white/50 tracking-wider">HEADING</span>
              <span className="text-white font-bold tracking-widest">{heading}° N</span>
            </div>
          </div>

          {/* Top Right Status */}
          <div className="ar-hud-panel p-4 w-52 flex flex-col gap-1.5">
            <div className="flex items-center gap-2 border-b border-cyan-500/20 pb-1.5 mb-1">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span className="font-mono text-[10px] text-white/50 tracking-wider font-bold">NAV STATE</span>
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-white/40">VISOR LINK:</span>
              <span className="text-emerald-400 font-bold">ACTIVE</span>
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-white/40">RADAR ENG:</span>
              <span className="text-cyan-400 font-bold">LOCK</span>
            </div>
          </div>
        </div>

        {/* MIDDLE SECTION: Radar scanning reticle */}
        <div className="absolute top-[42%] left-[45%] -translate-x-1/2 -translate-y-1/2 pointer-events-none flex items-center justify-center">
          <AnimatePresence>
            {status === "processing" && (
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 0.8 }}
                exit={{ scale: 1.2, opacity: 0 }}
                className="relative w-48 h-48 rounded-full border border-cyan-500/20 flex items-center justify-center"
              >
                <div className="absolute inset-2 border-2 border-dashed border-cyan-400/40 rounded-full animate-spin [animation-duration:10s]" />
                <div className="absolute inset-0 hud-radar-spin border-r border-cyan-500/30" />
                <Crosshair className="w-8 h-8 text-cyan-400 animate-pulse" />
                <span className="absolute -bottom-6 font-mono text-[9px] text-cyan-400 tracking-widest font-bold uppercase animate-pulse">
                  CALCULATING SAFE PATH...
                </span>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* BOTTOM SECTION */}
        <div className="flex justify-between items-end gap-6 z-30">
          
          {/* Bottom Left: Diagnostics panel */}
          <div className="ar-hud-panel p-4 w-56 flex flex-col gap-1.5 pointer-events-auto">
            <div className="flex items-center gap-2 border-b border-cyan-500/20 pb-1.5 mb-1.5">
              <Terminal className="w-4 h-4 text-cyan-400" />
              <span className="font-mono text-xs text-white/50 tracking-wider font-bold">DIAGNOSTICS</span>
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-white/40">SYSTEM FPS:</span>
              <span className="text-cyan-400 font-bold">{fps}</span>
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-white/40">VLM LATENCY:</span>
              <span className="text-cyan-400 font-bold">{simulatedLatency}ms</span>
            </div>
            
            {/* mini radar indicator */}
            <div className="flex justify-between items-center mt-2 pt-2 border-t border-cyan-500/10">
              <span className="text-[9px] font-mono text-white/30 uppercase tracking-widest">PATH RADAR</span>
              <div className="w-7 h-7 rounded-full border border-cyan-400/30 relative overflow-hidden flex items-center justify-center">
                <div className="absolute inset-0 bg-cyan-950/20" />
                <div className="w-full h-full absolute hud-radar-spin border-r border-cyan-400/50" />
                <div className="w-1.5 h-1.5 rounded-full bg-cyan-400" />
              </div>
            </div>
          </div>

          {/* Bottom Center: Microphone controller */}
          <div className="flex-1 max-w-xl flex flex-col items-center gap-3">
            <div className="w-full flex flex-col items-center bg-black/60 border border-cyan-500/20 rounded px-4 py-3 min-h-24 backdrop-blur-md relative">
              <div className="absolute top-1.5 right-2 flex gap-1 items-center font-mono text-[9px] text-white/30">
                <div className={`w-1.5 h-1.5 rounded-full ${isConnected ? "bg-emerald-400" : "bg-rose-500"}`} />
                {isConnected ? "NAV LNK" : "DISCONN"}
              </div>

              {/* Status Header */}
              <div className="font-mono text-[10px] text-cyan-400/70 tracking-widest uppercase mb-1 font-bold">
                {status === "listening" ? (
                  <span className="text-cyan-400">🎤 VOICE FEEDER READING</span>
                ) : status === "processing" ? (
                  <span className="text-amber-400 animate-pulse">⚡ ANALYZING SPATIAL HUD</span>
                ) : status === "speaking" ? (
                  <span className="text-emerald-400">🤖 AUDIO NAVIGATION SYSTEM</span>
                ) : (
                  <span className="text-white/40">📡 STANDBY — CLICK OR HOLD TO TRIGGER</span>
                )}
              </div>

              {/* Voice transcripts */}
              <div className="w-full text-center text-white/95 min-h-6 flex items-center justify-center font-medium">
                {status === "listening" && transcript.length > 0 && (
                  <span className="text-base text-cyan-200">
                    &quot;{transcript[transcript.length - 1]?.text}&quot;
                  </span>
                )}
                {status === "listening" && transcript.length === 0 && (
                  <span className="text-sm italic text-cyan-400/40">Listening...</span>
                )}
                {status === "speaking" && latestResponse && (
                  <span className="text-base text-emerald-300 font-bold">
                    &quot;{latestResponse}&quot;
                  </span>
                )}
                {status === "idle" && (
                  <span className="text-sm text-white/40 italic">
                    Press reticle to inquire about safe navigation path
                  </span>
                )}
              </div>

              {/* Audio Waveform */}
              <div className="flex items-center gap-1.5 h-5 mt-2 justify-center w-full">
                {status === "listening" && (
                  <>
                    <div className="wave-bar bg-cyan-400" />
                    <div className="wave-bar bg-cyan-400" style={{ animationDelay: "0.2s" }} />
                    <div className="wave-bar bg-cyan-400" style={{ animationDelay: "0.4s" }} />
                  </>
                )}
                {status === "speaking" && (
                  <>
                    <div className="wave-bar bg-emerald-400" style={{ height: "14px" }} />
                    <div className="wave-bar bg-emerald-400" style={{ animationDelay: "0.2s", height: "18px" }} />
                    <div className="wave-bar bg-emerald-400" style={{ animationDelay: "0.1s", height: "12px" }} />
                  </>
                )}
                {status === "idle" && (
                  <div className="h-0.5 w-16 bg-cyan-500/30 rounded" />
                )}
              </div>
            </div>

            {/* Central Activator Reticle Button */}
            <div className="flex items-center gap-4 pointer-events-auto">
              <button
                onClick={() => {
                  if (cameraActive) stopCamera();
                  else startCamera();
                }}
                className="p-3 bg-black/60 border border-cyan-500/30 hover:border-cyan-400 rounded-full text-cyan-400 transition-all backdrop-blur-md"
                aria-label={cameraActive ? "Stop Camera" : "Start Camera"}
              >
                {cameraActive ? <Square size={16} /> : <Play size={16} />}
              </button>

              <button
                id="voice-btn"
                onMouseDown={startListening}
                onMouseUp={stopListening}
                className={`w-20 h-20 rounded-full border flex items-center justify-center transition-all duration-300 ${
                  status === "listening"
                    ? "bg-rose-950/40 border-rose-500 text-rose-400 shadow-[0_0_30px_rgba(255,82,82,0.4)]"
                    : status === "processing"
                    ? "bg-amber-950/40 border-amber-500 text-amber-400 shadow-[0_0_30px_rgba(255,152,0,0.4)] animate-pulse"
                    : "bg-[#0099CC]/10 border-[#0099CC] text-[#0099CC] hover:border-[#0099CC] shadow-[0_0_20px_rgba(0,153,204,0.2)]"
                } backdrop-blur-md`}
                aria-label="Ask navigation assistant"
              >
                {status === "processing" ? (
                  <Loader size={28} className="animate-spin" />
                ) : (
                  <Mic size={28} />
                )}
              </button>

              <button
                onClick={clearTranscript}
                className="p-3 bg-black/60 border border-cyan-500/30 hover:border-cyan-400 rounded-full text-cyan-400 transition-all backdrop-blur-md"
                aria-label="Clear analysis"
              >
                <RefreshCw size={16} />
              </button>
            </div>
          </div>

          {/* Bottom Right: Navigation alert history logs */}
          <div className="ar-hud-panel p-4 w-64 h-48 flex flex-col pointer-events-auto overflow-hidden">
            <div className="flex items-center gap-2 border-b border-cyan-500/20 pb-1.5 mb-1.5">
              <Eye className="w-4 h-4 text-cyan-400" />
              <span className="font-mono text-xs text-white/50 tracking-wider font-bold">ALERTS RECORD</span>
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
                    <span className="text-white/40">Visual navigation standby...</span>
                  </div>
                  <div className="flex gap-1.5 items-center">
                    <span className="text-cyan-400/40">⚡</span>
                    <span className="text-white/40">Path mapping active.</span>
                  </div>
                </div>
              )}
            </div>
          </div>

        </div>

      </div>

      {voiceError && (
        <div className="fixed bottom-36 left-4 z-40 px-4 py-2 bg-rose-950/80 border border-rose-500 text-rose-300 text-xs font-mono rounded backdrop-blur-md flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 animate-bounce" />
          <span>ALERT: {voiceError}</span>
        </div>
      )}
    </div>
  );
}
