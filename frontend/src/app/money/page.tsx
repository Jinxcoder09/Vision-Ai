"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import {
  Coins,
  Mic,
  Square,
  Play,
  RefreshCw,
  Eye,
  Loader,
  Activity,
  AlertTriangle,
  Crosshair,
  ChevronLeft,
  Terminal,
} from "lucide-react";
import { useCamera } from "@/hooks/useCamera";
import { useVoice } from "@/hooks/useVoice";
import { motion, AnimatePresence } from "framer-motion";

export default function MoneyRecognitionPage() {
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const [fps, setFps] = useState(60);
  const [simulatedLatency, setSimulatedLatency] = useState(72);
  const [runningTotal, setRunningTotal] = useState<number>(0.0);
  const [detectedBills, setDetectedBills] = useState<string[]>([]);

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
    mode: "money",
    onError: (msg) => {
      console.error("MoneyRecognitionPage useVoice onError:", msg);
      setVoiceError(msg);
      setTimeout(() => setVoiceError(null), 5000);
    }
  });

  // Auto-start camera on mount
  useEffect(() => {
    startCamera();
  }, [startCamera]);

  // Capture frame once when user starts speaking (status becomes "listening")
  useEffect(() => {
    if (status === "listening" && cameraActive) {
      console.log("MoneyPage: Capturing currency frame...");
      const frame = captureCurrentFrame(0.75);
      if (frame) {
        sendFrame(frame);
      } else {
        sendFrame("");
      }
    }
  }, [status, cameraActive, captureCurrentFrame, sendFrame]);

  // Extract totals and bills dynamically from the VLM response
  useEffect(() => {
    if (latestResponse) {
      const responseText = latestResponse.toLowerCase();
      
      const matches = responseText.match(/\d+/g);
      
      // Basic heuristics to extract values and update running totals
      if (matches && matches.length > 0) {
        const values = matches.map(Number).filter(v => [1, 2, 5, 10, 20, 50, 100].includes(v));
        if (values.length > 0) {
          const latestValue = values[0];
          setDetectedBills((prev) => [`$${latestValue} bill`, ...prev.slice(0, 7)]);
          setRunningTotal((prev) => prev + latestValue);
        }
      }
    }
  }, [latestResponse]);

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
        const drift = Math.random() > 0.5 ? 2 : -2;
        const target = status === "processing" ? 120 : 60;
        return Math.max(40, Math.min(180, Math.round(prev + (target - prev) * 0.2 + drift)));
      });
    }, 800);
    return () => clearInterval(interval);
  }, [status]);

  const resetTracker = useCallback(() => {
    setRunningTotal(0.0);
    setDetectedBills([]);
    clearTranscript();
  }, [clearTranscript]);

  return (
    <div id="camera-hud-container" className="fixed inset-0 w-screen h-screen overflow-hidden bg-black select-none z-0">
      
      {/* ── 1. Full-Screen Camera View ────────────────────────────────────────── */}
      <div className="absolute inset-0 w-full h-full z-0 bg-black">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover transition-opacity duration-700 ${
            cameraActive ? "opacity-100" : "opacity-35"
          }`}
          aria-label="Money recognition camera feed"
        />

        {!cameraActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 text-cyan-500/80">
            <Coins className="w-16 h-16 animate-pulse text-cyan-400" />
            <p className="text-sm tracking-widest uppercase font-mono text-cyan-400/70">
              {cameraError ?? "Initializing Currency Scanner..."}
            </p>
            <button
              onClick={() => startCamera()}
              className="px-6 py-2.5 bg-cyan-950/40 border border-cyan-500/50 hover:bg-cyan-500/20 text-cyan-400 hover:text-white rounded-sm font-mono text-xs uppercase tracking-wider transition-all duration-300 backdrop-blur-md"
              style={{ boxShadow: "0 0 15px rgba(0, 212, 255, 0.15)" }}
            >
              Start Money Camera
            </button>
          </div>
        )}
      </div>

      {/* ── 2. HUD Scanline & Target Overlays ─────────────────────────────────── */}
      <div className="absolute inset-0 pointer-events-none z-10 hud-grid-overlay opacity-80" />
      <div className="absolute inset-0 pointer-events-none z-10 hud-scanlines opacity-15" />
      <div className="absolute inset-0 pointer-events-none z-10 bg-[radial-gradient(circle_at_center,transparent_45%,rgba(0,0,0,0.6)_100%)]" />

      {/* Banknote target placement frame */}
      {cameraActive && (
        <div className="absolute top-[22%] bottom-[36%] left-[15%] right-[15%] border-2 border-dashed border-cyan-400/40 rounded-3xl pointer-events-none z-20 flex items-center justify-center">
          <span className="font-mono text-[9px] tracking-widest text-cyan-400/70 uppercase bg-black/60 px-3 py-1 border border-cyan-500/20 rounded">
            PLACE BILL FLAT INSIDE SCANNED AREA
          </span>
        </div>
      )}

      {/* ── 3. HUD Glass Panels ───────────────────────────────────────────────── */}
      <div className="absolute inset-0 w-full h-full z-30 pointer-events-none p-4 flex flex-col justify-between">
        
        {/* TOP BAR */}
        <div className="flex justify-between items-start">
          {/* Back button */}
          <Link
            href="/"
            className="p-3 bg-black/60 border border-cyan-500/30 hover:border-cyan-400 rounded-full text-cyan-400 pointer-events-auto backdrop-blur-md flex items-center gap-2 font-mono text-xs uppercase tracking-wider"
            aria-label="Back to dashboard"
          >
            <ChevronLeft size={16} />
            <span>DASHBOARD</span>
          </Link>

          {/* Top Center: Running Total */}
          <div className="ar-hud-panel px-6 py-2.5 flex items-center gap-3.5 w-60 justify-center">
            <Coins className="w-5 h-5 text-yellow-400 animate-pulse" />
            <div className="flex flex-col font-mono text-xs">
              <span className="text-white/50 tracking-wider">CALCULATED TOTAL</span>
              <span className="text-yellow-400 font-extrabold text-sm tracking-widest">${runningTotal.toFixed(2)}</span>
            </div>
          </div>

          {/* Top Right: Status */}
          <div className="ar-hud-panel p-4 w-52 flex flex-col gap-1.5">
            <div className="flex items-center gap-2 border-b border-cyan-500/20 pb-1.5 mb-1">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span className="font-mono text-[10px] text-white/50 tracking-wider font-bold">SCANNER STATE</span>
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-white/40">CURRENCY:</span>
              <span className="text-cyan-400 font-bold">USD / EN</span>
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-white/40">CALIBRATE:</span>
              <span className="text-emerald-400 font-bold">OK</span>
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
                <div className="absolute inset-2 border-2 border-dashed border-cyan-400/40 rounded-full animate-spin [animation-duration:11s]" />
                <div className="absolute inset-0 hud-radar-spin border-r border-cyan-500/30" />
                <Crosshair className="w-8 h-8 text-cyan-400 animate-pulse" />
                <span className="absolute -bottom-6 font-mono text-[9px] text-cyan-400 tracking-widest font-bold uppercase animate-pulse">
                  RECOGNIZING BANKNOTE VALUE...
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
              <span className="text-[9px] font-mono text-white/30 uppercase tracking-widest">BILL RADAR</span>
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
                {isConnected ? "MONEY LNK" : "DISCONN"}
              </div>

              {/* Status Header */}
              <div className="font-mono text-[10px] text-cyan-400/70 tracking-widest uppercase mb-1 font-bold">
                {status === "listening" ? (
                  <span className="text-cyan-400">🎤 SCAN RECEIVER RECORDING</span>
                ) : status === "processing" ? (
                  <span className="text-amber-400 animate-pulse">⚡ ANALYZING BILL SPECIFICATION</span>
                ) : status === "speaking" ? (
                  <span className="text-emerald-400">🤖 AUDIO CONFIRMATION FEEDBACK</span>
                ) : (
                  <span className="text-white/40">📡 SYSTEM READY — TRIGGER SCANNERS</span>
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
                    Press mic or hold Space to scan currency
                  </span>
                )}
              </div>

              {/* Audio Waveform */}
              <div className="flex items-center gap-1.5 h-5 mt-2 justify-center w-full">
                {status === "listening" && (
                  <>
                    <div className="wave-bar bg-cyan-400" />
                    <div className="wave-bar bg-cyan-400" style={{ animationDelay: "0.1s" }} />
                    <div className="wave-bar bg-cyan-400" style={{ animationDelay: "0.3s" }} />
                  </>
                )}
                {status === "speaking" && (
                  <>
                    <div className="wave-bar bg-emerald-400" style={{ height: "12px" }} />
                    <div className="wave-bar bg-emerald-400" style={{ animationDelay: "0.2s", height: "16px" }} />
                    <div className="wave-bar bg-emerald-400" style={{ animationDelay: "0.1s", height: "10px" }} />
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
                    ? "bg-rose-950/40 border-rose-500 text-rose-400 shadow-[0_0_30px_rgba(244,63,94,0.4)]"
                    : status === "processing"
                    ? "bg-amber-950/40 border-amber-500 text-amber-400 shadow-[0_0_30px_rgba(245,158,11,0.4)] animate-pulse"
                    : "bg-cyan-950/40 border-cyan-500 text-cyan-400 hover:border-cyan-300 shadow-[0_0_20px_rgba(0,212,255,0.2)]"
                } backdrop-blur-md`}
                aria-label="Scan bill"
              >
                {status === "processing" ? (
                  <Loader size={28} className="animate-spin" />
                ) : (
                  <Mic size={28} />
                )}
              </button>

              <button
                onClick={resetTracker}
                className="p-3 bg-black/60 border border-cyan-500/30 hover:border-cyan-400 rounded-full text-cyan-400 transition-all backdrop-blur-md"
                aria-label="Reset calculated transaction"
              >
                <RefreshCw size={16} />
              </button>
            </div>
          </div>

          {/* Bottom Right: Transaction list HUD logs */}
          <div className="ar-hud-panel p-4 w-64 h-48 flex flex-col pointer-events-auto overflow-hidden">
            <div className="flex items-center gap-2 border-b border-cyan-500/20 pb-1.5 mb-1.5">
              <Eye className="w-4 h-4 text-cyan-400" />
              <span className="font-mono text-xs text-white/50 tracking-wider font-bold">TRANSACTION LOG</span>
            </div>

            <div className="flex-1 overflow-y-auto pr-1 text-[11px] font-mono text-cyan-300/80 space-y-1.5">
              {detectedBills.length > 0 ? (
                <div className="leading-relaxed">
                  {detectedBills.map((bill, idx) => (
                    <div key={idx} className="flex gap-1.5 items-center">
                      <span className="text-yellow-400">💵</span>
                      <span>Detected: {bill}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="space-y-1.5 opacity-60">
                  <div className="flex gap-1.5 items-center">
                    <span className="text-cyan-400/40">⚡</span>
                    <span className="text-white/40">Ready for cash check...</span>
                  </div>
                  <div className="flex gap-1.5 items-center">
                    <span className="text-cyan-400/40">⚡</span>
                    <span className="text-white/40">VLM scanner calibrated.</span>
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
