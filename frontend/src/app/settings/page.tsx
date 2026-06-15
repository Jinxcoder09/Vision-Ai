"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import {
  Camera,
  Volume2,
  Zap,
  RotateCcw,
  CheckCircle,
  ChevronLeft,
  Mic,
  Eye,
  Sliders,
} from "lucide-react";
import { useCamera } from "@/hooks/useCamera";
import { synthesizeSpeech } from "@/lib/api";
import { playWavBytes } from "@/lib/utils";

const VOICES = [
  { value: "af_heart", label: "Heart (American Female)", accent: "American" },
  { value: "af_bella", label: "Bella (American Female)", accent: "American" },
  { value: "af_nicole", label: "Nicole (American Female)", accent: "American" },
  { value: "am_adam", label: "Adam (American Male)", accent: "American" },
  { value: "am_michael", label: "Michael (American Male)", accent: "American" },
  { value: "bf_emma", label: "Emma (British Female)", accent: "British" },
  { value: "bf_isabella", label: "Isabella (British Female)", accent: "British" },
  { value: "bm_george", label: "George (British Male)", accent: "British" },
];

interface SettingRowProps {
  label: string;
  description?: string;
  id: string;
  children: React.ReactNode;
}

function SettingRow({ label, description, id, children }: SettingRowProps) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 py-5"
      style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
      <div className="flex-1">
        <label htmlFor={id} className="block font-bold text-white text-sm sm:text-base mb-1">
          {label}
        </label>
        {description && (
          <p className="text-slate-400 text-xs sm:text-sm font-medium">{description}</p>
        )}
      </div>
      <div className="flex-shrink-0 w-full sm:w-64">{children}</div>
    </div>
  );
}

export default function SettingsPage() {
  const { devices: cameras } = useCamera();
  const [micDevices, setMicDevices] = useState<MediaDeviceInfo[]>([]);

  // ── 1. Voice Settings ──
  const [voice, setVoice] = useState("af_heart");
  const [speed, setSpeed] = useState(1.0);
  const [pitch, setPitch] = useState(1.0);
  const [volume, setVolume] = useState(1.0);
  const [autoSpeak, setAutoSpeak] = useState(true);

  // ── 2. Microphone Settings ──
  const [micId, setMicId] = useState("");
  const [sensitivity, setSensitivity] = useState(0.5);
  const [autoListening, setAutoListening] = useState(true);
  const [vadThreshold, setVadThreshold] = useState(0.5);

  // ── 3. Camera Settings ──
  const [cameraId, setCameraId] = useState("");
  const [resolution, setResolution] = useState("720p");
  const [fps, setFps] = useState(30);

  // ── 4. Accessibility Settings ──
  const [textSize, setTextSize] = useState("normal");
  const [contrastMode, setContrastMode] = useState("normal");
  const [colorTheme, setColorTheme] = useState("default");
  const [screenReader, setScreenReader] = useState(false);
  const [voiceFirst, setVoiceFirst] = useState(true);

  // ── 5. AI Settings ──
  const [sceneRefresh, setSceneRefresh] = useState(2.5);
  const [memoryDuration, setMemoryDuration] = useState(30.0);
  const [verbosity, setVerbosity] = useState("normal");
  const [visionSensitivity, setVisionSensitivity] = useState(0.5);

  const [saved, setSaved] = useState(false);
  const [testingVoice, setTestingVoice] = useState(false);

  // Load saved settings
  useEffect(() => {
    const s = localStorage.getItem("eyeva-settings");
    if (s) {
      try {
        const p = JSON.parse(s);
        if (p.voice) setVoice(p.voice);
        if (p.speed) setSpeed(p.speed);
        if (p.pitch) setPitch(p.pitch);
        if (p.volume) setVolume(p.volume);
        if (p.autoSpeak !== undefined) setAutoSpeak(p.autoSpeak);

        if (p.micId) setMicId(p.micId);
        if (p.sensitivity !== undefined) setSensitivity(p.sensitivity);
        if (p.autoListening !== undefined) setAutoListening(p.autoListening);
        if (p.vadThreshold !== undefined) setVadThreshold(p.vadThreshold);

        if (p.cameraId) setCameraId(p.cameraId);
        if (p.resolution) setResolution(p.resolution);
        if (p.fps) setFps(p.fps);

        if (p.textSize) setTextSize(p.textSize);
        if (p.contrastMode) setContrastMode(p.contrastMode);
        if (p.colorTheme) setColorTheme(p.colorTheme);
        if (p.screenReader !== undefined) setScreenReader(p.screenReader);
        if (p.voiceFirst !== undefined) setVoiceFirst(p.voiceFirst);

        if (p.sceneRefresh !== undefined) setSceneRefresh(p.sceneRefresh);
        if (p.memoryDuration !== undefined) setMemoryDuration(p.memoryDuration);
        if (p.verbosity) setVerbosity(p.verbosity);
        if (p.visionSensitivity !== undefined) setVisionSensitivity(p.visionSensitivity);
      } catch {}
    }
  }, []);

  // Enumerate input microphones
  useEffect(() => {
    navigator.mediaDevices.enumerateDevices().then((devs) => {
      setMicDevices(devs.filter((d) => d.kind === "audioinput"));
    }).catch(() => {});
  }, []);

  const handleSave = useCallback(async () => {
    const payload = {
      voice, speed, pitch, volume, autoSpeak,
      micId, sensitivity, autoListening, vadThreshold,
      cameraId, resolution, fps,
      textSize, contrastMode, colorTheme, screenReader, voiceFirst,
      sceneRefresh, memoryDuration, verbosity, visionSensitivity
    };

    localStorage.setItem("eyeva-settings", JSON.stringify(payload));
    setSaved(true);

    // Apply root CSS classes dynamically on save
    document.documentElement.className = "scroll-smooth";
    if (textSize && textSize !== "normal") {
      document.documentElement.classList.add(`text-size-${textSize}`);
    }
    if (contrastMode === "high") {
      document.documentElement.classList.add("contrast-high");
    }
    if (colorTheme && colorTheme !== "default") {
      document.documentElement.classList.add(`theme-${colorTheme}`);
    }

    // Spoken confirmation to assist visually impaired user
    try {
      const wav = await synthesizeSpeech("Settings saved successfully.", voice, speed);
      await playWavBytes(wav);
    } catch {}

    setTimeout(() => setSaved(false), 2000);
  }, [
    voice, speed, pitch, volume, autoSpeak,
    micId, sensitivity, autoListening, vadThreshold,
    cameraId, resolution, fps,
    textSize, contrastMode, colorTheme, screenReader, voiceFirst,
    sceneRefresh, memoryDuration, verbosity, visionSensitivity
  ]);

  const handleReset = useCallback(() => {
    localStorage.removeItem("eyeva-settings");
    setVoice("af_heart");
    setSpeed(1.0);
    setPitch(1.0);
    setVolume(1.0);
    setAutoSpeak(true);
    setMicId("");
    setSensitivity(0.5);
    setAutoListening(true);
    setVadThreshold(0.5);
    setCameraId("");
    setResolution("720p");
    setFps(30);
    setTextSize("normal");
    setContrastMode("normal");
    setColorTheme("default");
    setScreenReader(false);
    setVoiceFirst(true);
    setSceneRefresh(2.5);
    setMemoryDuration(30.0);
    setVerbosity("normal");
    setVisionSensitivity(0.5);

    // Apply defaults instantly
    document.documentElement.className = "scroll-smooth";

    // Play feedback sound
    synthesizeSpeech("Settings reset completed.", "af_heart", 1.0)
      .then(playWavBytes)
      .catch(() => {});
  }, []);

  const testVoice = useCallback(async () => {
    setTestingVoice(true);
    try {
      const wav = await synthesizeSpeech(
        "Confirming synthesis configuration. Pitch and speed calibrated.",
        voice,
        speed
      );
      await playWavBytes(wav);
    } catch {
      // ignore
    } finally {
      setTestingVoice(false);
    }
  }, [voice, speed]);

  return (
    <div className="max-w-4xl mx-auto px-6 py-12">
      {/* Header */}
      <div className="flex items-center justify-between mb-8 border-b border-cyan-500/10 pb-4">
        <div>
          <h1 className="text-4xl font-extrabold mb-2 text-white flex items-center gap-2">
            <span className="gradient-text">System Configuration</span>
          </h1>
          <p className="text-slate-400 text-sm sm:text-base font-semibold">
            Customize voice parameters, microphone bounds, camera feeds, and high contrast themes.
          </p>
        </div>

        <Link
          href="/"
          className="p-3 bg-black/60 border border-cyan-500/30 hover:border-cyan-400 rounded-full text-cyan-400 flex items-center gap-2 font-mono text-xs uppercase tracking-wider"
          aria-label="Back to dashboard"
        >
          <ChevronLeft size={16} />
          <span>DASHBOARD</span>
        </Link>
      </div>

      {/* ── 1. Voice Settings ── */}
      <section className="glass-card p-6 mb-6" aria-labelledby="voice-settings-heading">
        <h2 id="voice-settings-heading" className="flex items-center gap-2.5 font-bold text-white text-lg border-b border-white/5 pb-3 mb-4">
          <Volume2 size={22} className="text-cyan-400" aria-hidden="true" />
          Voice Settings
        </h2>

        <SettingRow id="voice-select" label="Voice Persona" description="Choose speech gender and accent.">
          <select id="voice-select" value={voice} onChange={(e) => setVoice(e.target.value)} aria-label="Select voice font">
            {VOICES.map((v) => (
              <option key={v.value} value={v.value}>{v.label}</option>
            ))}
          </select>
        </SettingRow>

        <SettingRow id="speed-range" label={`Speech Speed: ${speed.toFixed(1)}x`} description="Calibrate narration pace.">
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-500 font-bold">0.5x</span>
            <input id="speed-range" type="range" min={0.5} max={2.0} step={0.1} value={speed} onChange={(e) => setSpeed(parseFloat(e.target.value))} />
            <span className="text-xs text-slate-500 font-bold">2.0x</span>
          </div>
        </SettingRow>

        <SettingRow id="pitch-range" label={`Speech Pitch: ${pitch.toFixed(1)}x`} description="Calibrate vocal pitch tone.">
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-500 font-bold">0.5x</span>
            <input id="pitch-range" type="range" min={0.5} max={1.5} step={0.1} value={pitch} onChange={(e) => setPitch(parseFloat(e.target.value))} />
            <span className="text-xs text-slate-500 font-bold">1.5x</span>
          </div>
        </SettingRow>

        <SettingRow id="volume-range" label={`Speech Volume: ${Math.round(volume * 100)}%`} description="Configure narrator amplitude.">
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-500 font-bold">0%</span>
            <input id="volume-range" type="range" min={0.0} max={1.0} step={0.05} value={volume} onChange={(e) => setVolume(parseFloat(e.target.value))} />
            <span className="text-xs text-slate-500 font-bold">100%</span>
          </div>
        </SettingRow>

        <SettingRow id="auto-speak-toggle" label="Auto Speak Aloud" description="Speak VLM answers immediately.">
          <select id="auto-speak-toggle" value={autoSpeak ? "true" : "false"} onChange={(e) => setAutoSpeak(e.target.value === "true")}>
            <option value="true">Enabled (Narration On)</option>
            <option value="false">Disabled (Text Only)</option>
          </select>
        </SettingRow>

        <div className="pt-4">
          <button id="test-voice-btn" onClick={testVoice} disabled={testingVoice} className="btn-secondary">
            {testingVoice ? "Narrating..." : "Test Voice Font"}
          </button>
        </div>
      </section>

      {/* ── 2. Microphone Settings ── */}
      <section className="glass-card p-6 mb-6" aria-labelledby="mic-settings-heading">
        <h2 id="mic-settings-heading" className="flex items-center gap-2.5 font-bold text-white text-lg border-b border-white/5 pb-3 mb-4">
          <Mic size={22} className="text-rose-400" aria-hidden="true" />
          Microphone Settings
        </h2>

        <SettingRow id="mic-select" label="Audio input device" description="Source for capturing your voice.">
          <select id="mic-select" value={micId} onChange={(e) => setMicId(e.target.value)}>
            <option value="">Default Microphone</option>
            {micDevices.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>{d.label || `Microphone ${d.deviceId.slice(0, 5)}`}</option>
            ))}
          </select>
        </SettingRow>

        <SettingRow id="sensitivity-range" label={`Input Sensitivity: ${Math.round(sensitivity * 100)}%`} description="Narration amplification.">
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-500">Low</span>
            <input id="sensitivity-range" type="range" min={0.0} max={1.0} step={0.05} value={sensitivity} onChange={(e) => setSensitivity(parseFloat(e.target.value))} />
            <span className="text-xs text-slate-500">High</span>
          </div>
        </SettingRow>

        <SettingRow id="auto-listen-toggle" label="Auto Listening" description="feeder keeps stream open continuous.">
          <select id="auto-listen-toggle" value={autoListening ? "true" : "false"} onChange={(e) => setAutoListening(e.target.value === "true")}>
            <option value="true">Continuous Conversation Mode</option>
            <option value="false">Push-to-Talk Mode</option>
          </select>
        </SettingRow>

        <SettingRow id="vad-threshold-range" label={`VAD Threshold: ${vadThreshold.toFixed(2)}`} description="Voice Activity Detection sensitivity (Lower is more sensitive).">
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-500">0.1</span>
            <input id="vad-threshold-range" type="range" min={0.1} max={0.9} step={0.05} value={vadThreshold} onChange={(e) => setVadThreshold(parseFloat(e.target.value))} />
            <span className="text-xs text-slate-500">0.9</span>
          </div>
        </SettingRow>
      </section>

      {/* ── 3. Camera Settings ── */}
      <section className="glass-card p-6 mb-6" aria-labelledby="camera-settings-heading">
        <h2 id="camera-settings-heading" className="flex items-center gap-2.5 font-bold text-white text-lg border-b border-white/5 pb-3 mb-4">
          <Camera size={22} className="text-violet-400" aria-hidden="true" />
          Camera Settings
        </h2>

        <SettingRow id="camera-select" label="Target Camera" description="Source for OCR and VLM frames.">
          <select id="camera-select" value={cameraId} onChange={(e) => setCameraId(e.target.value)}>
            <option value="">Default Camera</option>
            {cameras.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>{d.label || `Camera ${d.deviceId.slice(0, 5)}`}</option>
            ))}
          </select>
        </SettingRow>

        <SettingRow id="resolution-select" label="Optimal Resolution" description="Lower resolution increases VLM bandwidth speeds.">
          <select id="resolution-select" value={resolution} onChange={(e) => setResolution(e.target.value)}>
            <option value="720p">720p HD (Ideal)</option>
            <option value="1080p">1080p Full HD</option>
            <option value="480p">480p SD (Fastest)</option>
          </select>
        </SettingRow>

        <SettingRow id="fps-select" label="Target Frame Rate" description="Cap browser capturing rates.">
          <select id="fps-select" value={fps.toString()} onChange={(e) => setFps(parseInt(e.target.value))}>
            <option value="30">30 FPS (Recommended)</option>
            <option value="60">60 FPS (Ultra Smooth)</option>
            <option value="15">15 FPS (Save Battery)</option>
          </select>
        </SettingRow>
      </section>

      {/* ── 4. Accessibility Settings ── */}
      <section className="glass-card p-6 mb-6" aria-labelledby="access-settings-heading">
        <h2 id="access-settings-heading" className="flex items-center gap-2.5 font-bold text-white text-lg border-b border-white/5 pb-3 mb-4">
          <Eye size={22} className="text-emerald-400" aria-hidden="true" />
          Accessibility Settings
        </h2>

        <SettingRow id="text-size-select" label="Text Display Scale" description="Increase browser text size globally.">
          <select id="text-size-select" value={textSize} onChange={(e) => setTextSize(e.target.value)}>
            <option value="normal">Normal Text</option>
            <option value="large">Large Text</option>
            <option value="xlarge">Extra Large Text</option>
          </select>
        </SettingRow>

        <SettingRow id="contrast-mode-select" label="High Contrast Visor" description="Solid pitch blacks and bright neon guidelines.">
          <select id="contrast-mode-select" value={contrastMode} onChange={(e) => setContrastMode(e.target.value)}>
            <option value="normal">Default HUD aesthetics</option>
            <option value="high">Enabled (High Contrast On)</option>
          </select>
        </SettingRow>

        <SettingRow id="theme-select" label="Color HUD Themes" description="Choose color palette configurations.">
          <select id="theme-select" value={colorTheme} onChange={(e) => setColorTheme(e.target.value)}>
            <option value="default">Neon Cyan / Violet (Hologram)</option>
            <option value="cyberpunk">Cyberpunk (Magenta / Green)</option>
            <option value="hologram">True Hologram (Sleek Aqua)</option>
            <option value="classic">Slate Classic (Navy / Slate)</option>
          </select>
        </SettingRow>

        <SettingRow id="reader-mode-toggle" label="Screen Reader Support" description="Force strict ARIA element updates.">
          <select id="reader-mode-toggle" value={screenReader ? "true" : "false"} onChange={(e) => setScreenReader(e.target.value === "true")}>
            <option value="false">Disabled (Recommended with internal TTS)</option>
            <option value="true">Enabled (Use screen reader alerts)</option>
          </select>
        </SettingRow>

        <SettingRow id="voice-first-toggle" label="Voice-First Prompting" description="Eagerly start voice loop in HUD portals.">
          <select id="voice-first-toggle" value={voiceFirst ? "true" : "false"} onChange={(e) => setVoiceFirst(e.target.value === "true")}>
            <option value="true">Enabled (Autostart Narration loops)</option>
            <option value="false">Disabled (Manual activation needed)</option>
          </select>
        </SettingRow>
      </section>

      {/* ── 5. AI Settings ── */}
      <section className="glass-card p-6 mb-8" aria-labelledby="ai-settings-heading">
        <h2 id="ai-settings-heading" className="flex items-center gap-2.5 font-bold text-white text-lg border-b border-white/5 pb-3 mb-4">
          <Sliders size={22} className="text-yellow-400" aria-hidden="true" />
          AI & Scene Memory Settings
        </h2>

        <SettingRow id="scene-refresh-range" label={`Scene Refresh Interval: ${sceneRefresh}s`} description="Continuous VLM capturing rate limit intervals.">
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-500">1.0s</span>
            <input id="scene-refresh-range" type="range" min={1.0} max={5.0} step={0.5} value={sceneRefresh} onChange={(e) => setSceneRefresh(parseFloat(e.target.value))} />
            <span className="text-xs text-slate-500">5.0s</span>
          </div>
        </SettingRow>

        <SettingRow id="memory-duration-range" label={`Memory Retention: ${memoryDuration}s`} description="Cache duration for detected items in scene memory.">
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-500">10s</span>
            <input id="memory-duration-range" type="range" min={10} max={120} step={10} value={memoryDuration} onChange={(e) => setMemoryDuration(parseInt(e.target.value))} />
            <span className="text-xs text-slate-500">120s</span>
          </div>
        </SettingRow>

        <SettingRow id="verbosity-select" label="Speech Verbosity" description="Desired length of AI narration responses.">
          <select id="verbosity-select" value={verbosity} onChange={(e) => setVerbosity(e.target.value)}>
            <option value="normal">Normal (1-2 sentences)</option>
            <option value="concise">Concise (Single sentence warning)</option>
            <option value="detailed">Detailed (Detailed scene items description)</option>
          </select>
        </SettingRow>

        <SettingRow id="vision-sensitivity-range" label={`Vision Frame Sensitivity: ${Math.round(visionSensitivity * 100)}%`} description="Sensitivity for triggering image difference updates.">
          <div className="flex items-center gap-3">
            <span className="text-xs text-slate-500">0%</span>
            <input id="vision-sensitivity-range" type="range" min={0.0} max={1.0} step={0.05} value={visionSensitivity} onChange={(e) => setVisionSensitivity(parseFloat(e.target.value))} />
            <span className="text-xs text-slate-500">100%</span>
          </div>
        </SettingRow>
      </section>

      {/* Save action buttons */}
      <div className="flex gap-4" role="group" aria-label="Settings actions">
        <button id="save-settings-btn" onClick={handleSave} className="btn-primary flex-1">
          {saved ? (
            <>
              <CheckCircle size={18} />
              <span>Settings Saved!</span>
            </>
          ) : (
            <>
              <Zap size={18} />
              <span>Save Configurations</span>
            </>
          )}
        </button>
        <button id="reset-settings-btn" onClick={handleReset} className="btn-secondary px-6">
          <RotateCcw size={18} />
          <span>Reset Defaults</span>
        </button>
      </div>
    </div>
  );
}
