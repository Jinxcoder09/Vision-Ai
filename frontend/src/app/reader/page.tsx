"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import {
  BookOpen,
  Camera,
  Play,
  Copy,
  RefreshCw,
  Languages,
  Loader,
  CheckCircle,
  Volume2,
  ChevronLeft,
  Activity,
  Terminal,
} from "lucide-react";
import { useCamera } from "@/hooks/useCamera";
import { readText, synthesizeSpeech } from "@/lib/api";
import { playWavBytes } from "@/lib/utils";

type Lang = "en" | "hi";

export default function TextReaderPage() {
  const [lang, setLang] = useState<Lang>("en");
  const [extractedText, setExtractedText] = useState("");
  const [wordCount, setWordCount] = useState(0);
  const [latency, setLatency] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);


  const {
    videoRef,
    isActive: cameraActive,
    error: cameraError,
    startCamera,
    captureCurrentFrame,
  } = useCamera();

  // Auto-start camera on mount
  useEffect(() => {
    startCamera();
  }, [startCamera]);

  // ── Read Text ──────────────────────────────────────────────────────────────
  const handleRead = useCallback(async () => {
    setError(null);
    const frame = captureCurrentFrame(0.8);
    if (!frame) {
      setError("No camera frame. Please wait for camera initialization.");
      return;
    }

    setLoading(true);
    try {
      const result = await readText(frame, lang);
      setExtractedText(result.text);
      setWordCount(result.word_count);
      setLatency(result.latency_ms);
      
      // Auto-read aloud parsed text for voice-first friendliness
      if (result.text && result.text.trim()) {
        try {
          const wav = await synthesizeSpeech(result.text);
          await playWavBytes(wav);
        } catch {}
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "OCR text extraction failed.");
    } finally {
      setLoading(false);
    }
  }, [captureCurrentFrame, lang]);

  // ── Play Audio ─────────────────────────────────────────────────────────────
  const handlePlay = useCallback(async () => {
    if (!extractedText || playing) return;
    setPlaying(true);
    try {
      const wavBuffer = await synthesizeSpeech(extractedText);
      await playWavBytes(wavBuffer);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Audio synthesis failed.");
    } finally {
      setPlaying(false);
    }
  }, [extractedText, playing]);

  // ── Copy ───────────────────────────────────────────────────────────────────
  const handleCopy = useCallback(async () => {
    if (!extractedText) return;
    await navigator.clipboard.writeText(extractedText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [extractedText]);

  // Keyboard shortcut: Space = Trigger Read
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.code === "Space" && e.target === document.body) {
        e.preventDefault();
        handleRead();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [handleRead]);

  return (
    <div id="hud-fullscreen" className="fixed inset-0 w-screen h-screen overflow-hidden bg-black select-none z-0">
      
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
          aria-label="Text reader camera viewfinder"
        />

        {!cameraActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 text-cyan-500/80">
            <Camera className="w-16 h-16 animate-pulse text-cyan-400" />
            <p className="text-sm tracking-widest uppercase font-mono text-cyan-400/70">
              {cameraError ?? "Initializing OCR Viewfinder..."}
            </p>
            <button
              onClick={() => startCamera()}
              className="px-6 py-2.5 bg-cyan-950/40 border border-cyan-500/50 hover:bg-cyan-500/20 text-cyan-400 hover:text-white rounded-sm font-mono text-xs uppercase tracking-wider transition-all duration-300 backdrop-blur-md"
              style={{ boxShadow: "0 0 15px rgba(0, 212, 255, 0.15)" }}
            >
              Start Reader Camera
            </button>
          </div>
        )}
      </div>

      {/* ── 2. HUD Scanline & Target Overlays ─────────────────────────────────── */}
      <div className="absolute inset-0 pointer-events-none z-10 hud-grid-overlay opacity-80" />
      <div className="absolute inset-0 pointer-events-none z-10 hud-scanlines opacity-15" />
      <div className="absolute inset-0 pointer-events-none z-10 bg-[radial-gradient(circle_at_center,transparent_45%,rgba(0,0,0,0.6)_100%)]" />

      {/* Target framing box */}
      {cameraActive && (
        <div className="absolute top-[20%] bottom-[38%] left-[10%] right-[10%] border-2 border-dashed border-[#0099CC]/50 rounded-2xl pointer-events-none z-20 flex items-center justify-center">
          <span className="font-mono text-[10px] tracking-widest text-[#0099CC]/80 uppercase bg-black/60 px-3 py-1 border border-[#0099CC]/30 rounded">
            ALIGN TEXT INSIDE TARGET FRAME
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

          {/* Top Center: OCR Status */}
          <div className="ar-hud-panel px-4 py-2 flex items-center gap-3">
            <div className={`w-2 h-2 rounded-full ${cameraActive ? "bg-emerald-400 animate-pulse" : "bg-white/20"}`} />
            <span className="font-mono text-xs text-white/70">TEXT READER MODE</span>
          </div>

          {/* Top Right: Telemetry metrics */}
          <div className="ar-hud-panel p-4 w-52 flex flex-col gap-1.5">
            <div className="flex items-center gap-2 border-b border-cyan-500/20 pb-1.5 mb-1">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span className="font-mono text-[10px] text-white/50 tracking-wider font-bold">OCR METRICS</span>
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-white/40">WORDS READ:</span>
              <span className="text-cyan-400 font-bold">{wordCount}</span>
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-white/40">LATENCY:</span>
              <span className="text-cyan-400 font-bold">{latency ? `${latency.toFixed(0)}ms` : "--"}</span>
            </div>
          </div>
        </div>

        {/* BOTTOM SECTION */}
        <div className="flex justify-between items-end gap-6 z-30 pointer-events-auto">
          
          {/* Bottom Left: Instructions */}
          <div className="ar-hud-panel p-4 w-56 flex flex-col gap-1.5 font-mono text-[10px] text-white/60">
            <span className="font-bold border-b border-cyan-500/20 pb-1.5 mb-1.5 text-cyan-400">READER HOTKEYS</span>
            <span>• SPACE: Capture & OCR text</span>
            <span>• Hold text steady at 30cm</span>
            <span>• Voice will read it automatically</span>
          </div>

          {/* Bottom Center: Actions Controller */}
          <div className="flex-1 max-w-xl flex flex-col items-center gap-3">
            <div className="flex items-center gap-4 bg-black/60 border border-cyan-500/20 rounded-full px-5 py-3.5 backdrop-blur-md">
              {/* Language Selector */}
              <div className="flex items-center gap-1.5 border-r border-cyan-500/30 pr-4">
                <Languages size={18} className="text-cyan-400" />
                <select
                  value={lang}
                  onChange={(e) => setLang(e.target.value as Lang)}
                  className="bg-transparent border-none text-white text-xs outline-none cursor-pointer font-mono font-bold"
                  aria-label="Select OCR language"
                >
                  <option value="en">English</option>
                  <option value="hi">Hindi</option>
                </select>
              </div>

              {/* Main OCR Capture Trigger Button */}
              <button
                id="read-text-btn"
                onClick={handleRead}
                disabled={loading || !cameraActive}
                className={`flex items-center gap-2 px-6 py-2.5 rounded-full text-xs font-bold font-mono tracking-widest uppercase transition-all duration-300 ${
                  loading
                    ? "bg-amber-950/40 border border-amber-500 text-amber-400"
                    : "bg-cyan-950/40 border border-cyan-500 hover:border-cyan-300 text-cyan-400"
                }`}
                aria-label="Scan and read text"
              >
                {loading ? (
                  <>
                    <Loader size={14} className="animate-spin" />
                    <span>SCANNING…</span>
                  </>
                ) : (
                  <>
                    <BookOpen size={14} />
                    <span>CAPTURE & READ</span>
                  </>
                )}
              </button>

              {/* Speech Playback Trigger */}
              <button
                id="play-text-btn"
                onClick={handlePlay}
                disabled={!extractedText || playing}
                className={`p-2.5 border rounded-full transition-all ${
                  playing
                    ? "border-emerald-500 text-emerald-400 bg-emerald-950/20"
                    : "border-cyan-500/30 text-cyan-400 hover:border-cyan-400"
                }`}
                aria-label="Read text aloud"
              >
                {playing ? <Volume2 size={16} className="animate-pulse" /> : <Play size={16} />}
              </button>

              {/* Copy Output Button */}
              <button
                id="copy-text-btn"
                onClick={handleCopy}
                disabled={!extractedText}
                className="p-2.5 border border-cyan-500/30 hover:border-cyan-400 rounded-full text-cyan-400 transition-all"
                aria-label="Copy scanned text"
              >
                {copied ? <CheckCircle size={16} className="text-emerald-400" /> : <Copy size={16} />}
              </button>

              {/* Reset Clear Trigger */}
              <button
                id="clear-text-btn"
                onClick={() => { setExtractedText(""); setWordCount(0); setLatency(null); setError(null); }}
                disabled={!extractedText}
                className="p-2.5 border border-cyan-500/30 hover:border-cyan-400 rounded-full text-cyan-400 transition-all"
                aria-label="Reset text reader"
              >
                <RefreshCw size={16} />
              </button>
            </div>
          </div>

          {/* Bottom Right: Extracted Text HUD Display */}
          <div className="ar-hud-panel p-4 w-72 h-52 flex flex-col overflow-hidden">
            <div className="flex items-center justify-between border-b border-cyan-500/20 pb-1.5 mb-1.5">
              <div className="flex items-center gap-2">
                <Terminal className="w-4 h-4 text-cyan-400" />
                <span className="font-mono text-xs text-white/50 tracking-wider font-bold">EXTRACTED CONTENT</span>
              </div>
            </div>

            <div className="flex-1 overflow-y-auto pr-1 text-xs font-mono text-cyan-300 leading-relaxed">
              {error ? (
                <span className="text-rose-400">⚠ ERROR: {error}</span>
              ) : extractedText ? (
                <span className="text-white bg-black/40 p-1.5 rounded block border border-cyan-500/10">
                  {extractedText}
                </span>
              ) : (
                <span className="text-white/30 italic block text-center mt-6">
                  Ready to scan packages, documents, or labels...
                </span>
              )}
            </div>
          </div>

        </div>

      </div>

    </div>
  );
}
