"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import {
  Volume2,
  Mic,
  Camera,
  Eye,
  Sliders,
  Palette,
  Shield,
  ChevronDown,
  ChevronUp,
  CheckCircle,
  RotateCcw,
  Zap,
  Play,
  Loader,
  AlertTriangle,
  Trash2,
  X,
} from "lucide-react";
import { useCamera } from "@/hooks/useCamera";
import { synthesizeSpeech } from "@/lib/api";
import { playWavBytes } from "@/lib/utils";

// ── Constants ─────────────────────────────────────────────────────────────────

const VOICES = [
  { value: "af_heart",    label: "Heart (American Female)",    accent: "American" },
  { value: "af_bella",    label: "Bella (American Female)",    accent: "American" },
  { value: "af_nicole",   label: "Nicole (American Female)",   accent: "American" },
  { value: "am_adam",     label: "Adam (American Male)",       accent: "American" },
  { value: "am_michael",  label: "Michael (American Male)",    accent: "American" },
  { value: "bf_emma",     label: "Emma (British Female)",      accent: "British"  },
  { value: "bf_isabella", label: "Isabella (British Female)",  accent: "British"  },
  { value: "bm_george",   label: "George (British Male)",      accent: "British"  },
];

// ── Sub-components ────────────────────────────────────────────────────────────

/** Generic setting row — label left, control right */
function SettingRow({
  id,
  label,
  description,
  children,
}: {
  id: string;
  label: string;
  description?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="setting-row">
      <div className="setting-label-wrap">
        <label htmlFor={id} className="setting-label">{label}</label>
        {description && <p className="setting-desc">{description}</p>}
      </div>
      <div className="setting-control">{children}</div>
    </div>
  );
}

/** Animated toggle switch — replaces boolean selects */
function Toggle({
  id,
  checked,
  onChange,
  labelOn = "Enabled",
  labelOff = "Disabled",
}: {
  id: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  labelOn?: string;
  labelOff?: string;
}) {
  return (
    <div className="toggle-wrapper">
      <label className="toggle-switch" aria-label={checked ? labelOn : labelOff}>
        <input
          id={id}
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
        />
        <span className="toggle-track" />
      </label>
      <span className="toggle-label">{checked ? labelOn : labelOff}</span>
    </div>
  );
}

/** Slider with gradient fill track and live value */
function Slider({
  id,
  min,
  max,
  step,
  value,
  onChange,
  format,
  minLabel,
  maxLabel,
}: {
  id: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  minLabel?: string;
  maxLabel?: string;
}) {
  const pct = ((value - min) / (max - min)) * 100;
  const fmt = format ?? ((v: number) => String(v));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{minLabel ?? String(min)}</span>
        <span
          style={{
            fontSize: 12,
            fontWeight: 700,
            color: "var(--accent-primary)",
            fontFamily: "var(--font-mono)",
            minWidth: 48,
            textAlign: "center",
          }}
        >
          {fmt(value)}
        </span>
        <span style={{ fontSize: 12, color: "var(--text-muted)" }}>{maxLabel ?? String(max)}</span>
      </div>
      <div style={{ position: "relative", height: 4 }}>
        {/* Track fill */}
        <div
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            height: "100%",
            width: `${pct}%`,
            background: "var(--accent-gradient)",
            borderRadius: 2,
            pointerEvents: "none",
          }}
        />
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          style={{ position: "absolute", inset: 0, background: "transparent" }}
        />
      </div>
    </div>
  );
}

/** Collapsible section wrapper */
function Section({
  id,
  title,
  icon,
  children,
  defaultOpen = true,
}: {
  id: string;
  title: string;
  icon: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="settings-section" id={id}>
      <button
        className="settings-section-header w-full text-left"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-controls={`${id}-body`}
        type="button"
      >
        <div className="settings-section-title">
          {icon}
          {title}
        </div>
        {open ? <ChevronUp size={16} style={{ color: "var(--text-muted)" }} /> : <ChevronDown size={16} style={{ color: "var(--text-muted)" }} />}
      </button>
      {open && (
        <div className="settings-section-body" id={`${id}-body`} role="group">
          {children}
        </div>
      )}
    </div>
  );
}

/** Confirmation Modal */
function ConfirmModal({
  title,
  message,
  confirmLabel,
  onConfirm,
  onCancel,
}: {
  title: string;
  message: string;
  confirmLabel: string;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="modal-title">
      <div className="modal">
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <AlertTriangle size={20} style={{ color: "var(--accent-warning)", flexShrink: 0 }} aria-hidden="true" />
            <h2 id="modal-title" style={{ fontWeight: 700, fontSize: 16, color: "var(--text-primary)" }}>{title}</h2>
          </div>
          <button className="btn-icon" onClick={onCancel} aria-label="Close dialog">
            <X size={18} />
          </button>
        </div>
        <p style={{ fontSize: 14, color: "var(--text-secondary)", marginBottom: 24, lineHeight: 1.6 }}>{message}</p>
        <div style={{ display: "flex", gap: 12, justifyContent: "flex-end" }}>
          <button className="btn-ghost" onClick={onCancel}>Cancel</button>
          <button className="btn-danger" onClick={onConfirm}>{confirmLabel}</button>
        </div>
      </div>
    </div>
  );
}

// ── Settings Page ─────────────────────────────────────────────────────────────

export default function SettingsPage() {
  const { devices: cameras } = useCamera();
  const [micDevices, setMicDevices] = useState<MediaDeviceInfo[]>([]);

  // ── 1. Voice Settings ──
  const [voice, setVoice]           = useState("af_heart");
  const [speed, setSpeed]           = useState(1.0);
  const [pitch, setPitch]           = useState(1.0);
  const [volume, setVolume]         = useState(1.0);
  const [autoSpeak, setAutoSpeak]   = useState(true);

  // ── 2. Microphone Settings ──
  const [micId, setMicId]                   = useState("");
  const [sensitivity, setSensitivity]       = useState(0.5);
  const [autoListening, setAutoListening]   = useState(true);
  const [vadThreshold, setVadThreshold]     = useState(0.5);

  // ── 3. Camera Settings ──
  const [cameraId, setCameraId]     = useState("");
  const [resolution, setResolution] = useState("720p");
  const [fps, setFps]               = useState(30);

  // ── 4. Accessibility Settings ──
  const [textSize, setTextSize]           = useState("normal");
  const [contrastMode, setContrastMode]   = useState("normal");
  const [screenReader, setScreenReader]   = useState(false);
  const [voiceFirst, setVoiceFirst]       = useState(true);

  // ── 5. AI Settings ──
  const [sceneRefresh, setSceneRefresh]         = useState(2.5);
  const [memoryDuration, setMemoryDuration]     = useState(30.0);
  const [verbosity, setVerbosity]               = useState("normal");
  const [visionSensitivity, setVisionSensitivity] = useState(0.5);

  // ── 6. Display / Theme ──
  const [colorTheme, setColorTheme] = useState("default");

  // ── UI State ──
  const [saved, setSaved]           = useState(false);
  const [testingVoice, setTestingVoice] = useState(false);
  const [showResetModal, setShowResetModal] = useState(false);
  const [cacheClearDone, setCacheClearDone] = useState(false);

  // ── Load saved settings ────────────────────────────────────────────────────
  useEffect(() => {
    const s = localStorage.getItem("eyeva-settings");
    if (s) {
      try {
        const p = JSON.parse(s);
        if (p.voice)              setVoice(p.voice);
        if (p.speed)              setSpeed(p.speed);
        if (p.pitch)              setPitch(p.pitch);
        if (p.volume)             setVolume(p.volume);
        if (p.autoSpeak !== undefined) setAutoSpeak(p.autoSpeak);
        if (p.micId)              setMicId(p.micId);
        if (p.sensitivity !== undefined) setSensitivity(p.sensitivity);
        if (p.autoListening !== undefined) setAutoListening(p.autoListening);
        if (p.vadThreshold !== undefined) setVadThreshold(p.vadThreshold);
        if (p.cameraId)           setCameraId(p.cameraId);
        if (p.resolution)         setResolution(p.resolution);
        if (p.fps)                setFps(p.fps);
        if (p.textSize)           setTextSize(p.textSize);
        if (p.contrastMode)       setContrastMode(p.contrastMode);
        if (p.colorTheme)         setColorTheme(p.colorTheme);
        if (p.screenReader !== undefined) setScreenReader(p.screenReader);
        if (p.voiceFirst !== undefined)   setVoiceFirst(p.voiceFirst);
        if (p.sceneRefresh !== undefined) setSceneRefresh(p.sceneRefresh);
        if (p.memoryDuration !== undefined) setMemoryDuration(p.memoryDuration);
        if (p.verbosity)          setVerbosity(p.verbosity);
        if (p.visionSensitivity !== undefined) setVisionSensitivity(p.visionSensitivity);
      } catch {}
    }
  }, []);

  // ── Enumerate mic devices ─────────────────────────────────────────────────
  useEffect(() => {
    navigator.mediaDevices.enumerateDevices().then((devs) => {
      setMicDevices(devs.filter((d) => d.kind === "audioinput"));
    }).catch(() => {});
  }, []);

  // ── Save ──────────────────────────────────────────────────────────────────
  const handleSave = useCallback(async () => {
    const payload = {
      voice, speed, pitch, volume, autoSpeak,
      micId, sensitivity, autoListening, vadThreshold,
      cameraId, resolution, fps,
      textSize, contrastMode, colorTheme, screenReader, voiceFirst,
      sceneRefresh, memoryDuration, verbosity, visionSensitivity,
    };
    localStorage.setItem("eyeva-settings", JSON.stringify(payload));
    setSaved(true);

    // Apply CSS classes immediately
    document.documentElement.className = "scroll-smooth";
    if (textSize && textSize !== "normal")
      document.documentElement.classList.add(`text-size-${textSize}`);
    if (contrastMode === "high")
      document.documentElement.classList.add("contrast-high");
    if (colorTheme && colorTheme !== "default")
      document.documentElement.classList.add(`theme-${colorTheme}`);

    try {
      const wav = await synthesizeSpeech("Settings saved successfully.", voice, speed);
      await playWavBytes(wav);
    } catch {}

    setTimeout(() => setSaved(false), 2500);
  }, [
    voice, speed, pitch, volume, autoSpeak,
    micId, sensitivity, autoListening, vadThreshold,
    cameraId, resolution, fps,
    textSize, contrastMode, colorTheme, screenReader, voiceFirst,
    sceneRefresh, memoryDuration, verbosity, visionSensitivity,
  ]);

  // ── Reset ─────────────────────────────────────────────────────────────────
  const handleReset = useCallback(() => {
    localStorage.removeItem("eyeva-settings");
    setVoice("af_heart"); setSpeed(1.0); setPitch(1.0); setVolume(1.0); setAutoSpeak(true);
    setMicId(""); setSensitivity(0.5); setAutoListening(true); setVadThreshold(0.5);
    setCameraId(""); setResolution("720p"); setFps(30);
    setTextSize("normal"); setContrastMode("normal"); setColorTheme("default");
    setScreenReader(false); setVoiceFirst(true);
    setSceneRefresh(2.5); setMemoryDuration(30.0); setVerbosity("normal"); setVisionSensitivity(0.5);
    document.documentElement.className = "scroll-smooth";
    setShowResetModal(false);
    synthesizeSpeech("Settings reset to defaults.", "af_heart", 1.0).then(playWavBytes).catch(() => {});
  }, []);

  // ── Test Voice ────────────────────────────────────────────────────────────
  const testVoice = useCallback(async () => {
    setTestingVoice(true);
    try {
      const wav = await synthesizeSpeech("Confirming synthesis. Pitch and speed calibrated.", voice, speed);
      await playWavBytes(wav);
    } catch {}
    finally { setTestingVoice(false); }
  }, [voice, speed]);

  // ── Clear Cache ───────────────────────────────────────────────────────────
  const handleClearCache = useCallback(() => {
    try {
      const keys = Object.keys(localStorage).filter(k => k !== "eyeva-settings");
      keys.forEach(k => localStorage.removeItem(k));
      sessionStorage.clear();
    } catch {}
    setCacheClearDone(true);
    setTimeout(() => setCacheClearDone(false), 2500);
  }, []);

  // ── Color theme preview ───────────────────────────────────────────────────
  const themeAccents: Record<string, { primary: string; secondary: string }> = {
    default:   { primary: "#0099CC", secondary: "#6A5ACD" },
    cyberpunk: { primary: "#ff007f", secondary: "#00f0ff" },
    hologram:  { primary: "#80f0ff", secondary: "#a0c0ff" },
    classic:   { primary: "#3b82f6", secondary: "#8b5cf6" },
  };

  return (
    <div style={{ maxWidth: 860, margin: "0 auto", padding: "40px 24px 48px" }}>

      {/* ── Page Header ──────────────────────────────────────────────────── */}
      <div style={{ marginBottom: 32 }}>
        <h1
          style={{
            fontSize: "clamp(24px, 4vw, 34px)",
            fontWeight: 800,
            letterSpacing: "-0.02em",
            color: "var(--text-primary)",
            marginBottom: 8,
          }}
        >
          <span className="gradient-text">System Configuration</span>
        </h1>
        <p style={{ fontSize: 14, color: "var(--text-secondary)" }}>
          Customize voice, microphone, camera, accessibility, and AI preferences.
        </p>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          1. VOICE SETTINGS
      ══════════════════════════════════════════════════════════════════ */}
      <Section
        id="voice-settings"
        title="Voice Settings"
        icon={<Volume2 size={18} style={{ color: "var(--accent-primary)" }} />}
        defaultOpen={true}
      >
        <SettingRow id="voice-select" label="Voice Persona" description="Choose speech gender and accent.">
          <select id="voice-select" value={voice} onChange={(e) => setVoice(e.target.value)}>
            {VOICES.map((v) => (
              <option key={v.value} value={v.value}>{v.label}</option>
            ))}
          </select>
        </SettingRow>

        <SettingRow id="speed-range" label="Speech Speed" description="Calibrate narration pace.">
          <Slider
            id="speed-range"
            min={0.5} max={2.0} step={0.1} value={speed}
            onChange={setSpeed}
            format={(v) => `${v.toFixed(1)}×`}
            minLabel="0.5×" maxLabel="2.0×"
          />
        </SettingRow>

        <SettingRow id="pitch-range" label="Speech Pitch" description="Calibrate vocal pitch tone.">
          <Slider
            id="pitch-range"
            min={0.5} max={1.5} step={0.1} value={pitch}
            onChange={setPitch}
            format={(v) => `${v.toFixed(1)}×`}
            minLabel="0.5×" maxLabel="1.5×"
          />
        </SettingRow>

        <SettingRow id="volume-range" label="Speech Volume" description="Configure narrator amplitude.">
          <Slider
            id="volume-range"
            min={0.0} max={1.0} step={0.05} value={volume}
            onChange={setVolume}
            format={(v) => `${Math.round(v * 100)}%`}
            minLabel="0%" maxLabel="100%"
          />
        </SettingRow>

        <SettingRow id="auto-speak-toggle" label="Auto Speak" description="Speak AI answers immediately on response.">
          <Toggle
            id="auto-speak-toggle"
            checked={autoSpeak}
            onChange={setAutoSpeak}
            labelOn="Narration On"
            labelOff="Text Only"
          />
        </SettingRow>

        {/* Test voice button */}
        <div style={{ padding: "16px 0 4px" }}>
          <button
            id="test-voice-btn"
            onClick={testVoice}
            disabled={testingVoice}
            className="btn-secondary"
            style={{ width: "100%" }}
          >
            {testingVoice ? (
              <><Loader size={16} className="animate-spin" aria-hidden="true" /> Testing voice…</>
            ) : (
              <><Play size={16} aria-hidden="true" /> Test Voice Font</>
            )}
          </button>
        </div>
      </Section>

      {/* ══════════════════════════════════════════════════════════════════
          2. MICROPHONE SETTINGS
      ══════════════════════════════════════════════════════════════════ */}
      <Section
        id="mic-settings"
        title="Microphone Settings"
        icon={<Mic size={18} style={{ color: "var(--accent-error)" }} />}
        defaultOpen={true}
      >
        <SettingRow id="mic-select" label="Audio Input Device" description="Source for capturing voice.">
          <select id="mic-select" value={micId} onChange={(e) => setMicId(e.target.value)}>
            <option value="">Default Microphone</option>
            {micDevices.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label || `Microphone ${d.deviceId.slice(0, 6)}`}
              </option>
            ))}
          </select>
        </SettingRow>

        <SettingRow id="sensitivity-range" label="Input Sensitivity" description="Amplification of incoming audio.">
          <Slider
            id="sensitivity-range"
            min={0.0} max={1.0} step={0.05} value={sensitivity}
            onChange={setSensitivity}
            format={(v) => `${Math.round(v * 100)}%`}
            minLabel="Low" maxLabel="High"
          />
        </SettingRow>

        <SettingRow id="auto-listen-toggle" label="Auto Listening" description="Keep stream open continuously.">
          <Toggle
            id="auto-listen-toggle"
            checked={autoListening}
            onChange={setAutoListening}
            labelOn="Continuous Mode"
            labelOff="Push-to-Talk"
          />
        </SettingRow>

        <SettingRow id="vad-range" label="VAD Threshold" description="Voice Activity Detection sensitivity. Lower = more sensitive.">
          <Slider
            id="vad-range"
            min={0.1} max={0.9} step={0.05} value={vadThreshold}
            onChange={setVadThreshold}
            format={(v) => v.toFixed(2)}
            minLabel="0.1" maxLabel="0.9"
          />
        </SettingRow>
      </Section>

      {/* ══════════════════════════════════════════════════════════════════
          3. CAMERA SETTINGS
      ══════════════════════════════════════════════════════════════════ */}
      <Section
        id="camera-settings"
        title="Camera Settings"
        icon={<Camera size={18} style={{ color: "var(--accent-secondary)" }} />}
        defaultOpen={false}
      >
        <SettingRow id="camera-select" label="Target Camera" description="Source for OCR and VLM frames.">
          <select id="camera-select" value={cameraId} onChange={(e) => setCameraId(e.target.value)}>
            <option value="">Default Camera</option>
            {cameras.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label || `Camera ${d.deviceId.slice(0, 6)}`}
              </option>
            ))}
          </select>
        </SettingRow>

        <SettingRow id="resolution-select" label="Capture Resolution" description="Lower resolution increases VLM bandwidth speed.">
          <select id="resolution-select" value={resolution} onChange={(e) => setResolution(e.target.value)}>
            <option value="480p">480p SD — Fastest</option>
            <option value="720p">720p HD — Ideal</option>
            <option value="1080p">1080p Full HD</option>
          </select>
        </SettingRow>

        <SettingRow id="fps-select" label="Target Frame Rate" description="Cap browser capture rate.">
          <select id="fps-select" value={String(fps)} onChange={(e) => setFps(parseInt(e.target.value))}>
            <option value="15">15 FPS — Battery Saver</option>
            <option value="30">30 FPS — Recommended</option>
            <option value="60">60 FPS — Ultra Smooth</option>
          </select>
        </SettingRow>
      </Section>

      {/* ══════════════════════════════════════════════════════════════════
          4. ACCESSIBILITY SETTINGS
      ══════════════════════════════════════════════════════════════════ */}
      <Section
        id="access-settings"
        title="Accessibility"
        icon={<Eye size={18} style={{ color: "var(--accent-success)" }} />}
        defaultOpen={false}
      >
        <SettingRow id="text-size-select" label="Text Display Scale" description="Increase text size globally across the app.">
          <select id="text-size-select" value={textSize} onChange={(e) => setTextSize(e.target.value)}>
            <option value="normal">Normal</option>
            <option value="large">Large</option>
            <option value="xlarge">Extra Large</option>
          </select>
        </SettingRow>

        <SettingRow id="contrast-select" label="High Contrast Visor" description="Solid blacks and bright high-contrast guidelines.">
          <select id="contrast-select" value={contrastMode} onChange={(e) => setContrastMode(e.target.value)}>
            <option value="normal">Default Aesthetics</option>
            <option value="high">High Contrast On</option>
          </select>
        </SettingRow>

        <SettingRow id="screen-reader-toggle" label="Screen Reader Support" description="Force strict ARIA element updates for external readers.">
          <Toggle
            id="screen-reader-toggle"
            checked={screenReader}
            onChange={setScreenReader}
            labelOn="Screen Reader Alerts"
            labelOff="Internal TTS Only"
          />
        </SettingRow>

        <SettingRow id="voice-first-toggle" label="Voice-First Prompting" description="Auto-start voice loop when entering HUD portals.">
          <Toggle
            id="voice-first-toggle"
            checked={voiceFirst}
            onChange={setVoiceFirst}
            labelOn="Autostart On"
            labelOff="Manual Activation"
          />
        </SettingRow>
      </Section>

      {/* ══════════════════════════════════════════════════════════════════
          5. AI SETTINGS
      ══════════════════════════════════════════════════════════════════ */}
      <Section
        id="ai-settings"
        title="AI & Scene Memory"
        icon={<Sliders size={18} style={{ color: "var(--accent-warning)" }} />}
        defaultOpen={false}
      >
        <SettingRow id="scene-refresh-range" label="Scene Refresh Interval" description="Continuous VLM frame capture rate.">
          <Slider
            id="scene-refresh-range"
            min={1.0} max={5.0} step={0.5} value={sceneRefresh}
            onChange={setSceneRefresh}
            format={(v) => `${v.toFixed(1)}s`}
            minLabel="1.0s" maxLabel="5.0s"
          />
        </SettingRow>

        <SettingRow id="memory-range" label="Memory Retention" description="Cache duration for detected objects in scene memory.">
          <Slider
            id="memory-range"
            min={10} max={120} step={10} value={memoryDuration}
            onChange={setMemoryDuration}
            format={(v) => `${v}s`}
            minLabel="10s" maxLabel="120s"
          />
        </SettingRow>

        <SettingRow id="verbosity-select" label="Speech Verbosity" description="Desired length of AI narration responses.">
          <select id="verbosity-select" value={verbosity} onChange={(e) => setVerbosity(e.target.value)}>
            <option value="concise">Concise — Single sentence</option>
            <option value="normal">Normal — 1-2 sentences</option>
            <option value="detailed">Detailed — Full scene description</option>
          </select>
        </SettingRow>

        <SettingRow id="vision-range" label="Vision Sensitivity" description="Threshold for triggering image-difference updates.">
          <Slider
            id="vision-range"
            min={0.0} max={1.0} step={0.05} value={visionSensitivity}
            onChange={setVisionSensitivity}
            format={(v) => `${Math.round(v * 100)}%`}
            minLabel="0%" maxLabel="100%"
          />
        </SettingRow>
      </Section>

      {/* ══════════════════════════════════════════════════════════════════
          6. DISPLAY / THEME
      ══════════════════════════════════════════════════════════════════ */}
      <Section
        id="theme-settings"
        title="Display & Theme"
        icon={<Palette size={18} style={{ color: "#A855F7" }} />}
        defaultOpen={false}
      >
        <SettingRow id="theme-select" label="Color Theme" description="Choose your HUD color palette.">
          <select id="theme-select" value={colorTheme} onChange={(e) => setColorTheme(e.target.value)}>
            <option value="default">Professional Cyan / Slate (Default)</option>
            <option value="cyberpunk">Cyberpunk — Magenta / Cyan</option>
            <option value="hologram">Hologram — Soft Aqua / Blue</option>
            <option value="classic">Classic — Navy / Slate Blue</option>
          </select>
        </SettingRow>

        {/* Theme preview swatches */}
        <div style={{ padding: "12px 0 16px", display: "flex", gap: 10, flexWrap: "wrap" }}>
          {Object.entries(themeAccents).map(([key, { primary, secondary }]) => (
            <button
              key={key}
              type="button"
              onClick={() => setColorTheme(key)}
              aria-label={`Select ${key} theme`}
              aria-pressed={colorTheme === key}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 6,
                padding: "10px 14px",
                borderRadius: 8,
                border: `2px solid ${colorTheme === key ? primary : "var(--border-default)"}`,
                background: colorTheme === key ? `${primary}12` : "transparent",
                cursor: "pointer",
                transition: "all 0.15s ease",
              }}
            >
              <div style={{ display: "flex", gap: 3 }}>
                <div style={{ width: 14, height: 14, borderRadius: "50%", background: primary }} />
                <div style={{ width: 14, height: 14, borderRadius: "50%", background: secondary }} />
              </div>
              <span style={{ fontSize: 10, color: "var(--text-secondary)", fontWeight: 500, textTransform: "capitalize" }}>
                {key}
              </span>
            </button>
          ))}
        </div>
      </Section>

      {/* ══════════════════════════════════════════════════════════════════
          7. ADVANCED
      ══════════════════════════════════════════════════════════════════ */}
      <Section
        id="advanced-settings"
        title="Advanced"
        icon={<Shield size={18} style={{ color: "var(--text-muted)" }} />}
        defaultOpen={false}
      >
        {/* Clear App Cache */}
        <div className="setting-row">
          <div className="setting-label-wrap">
            <span className="setting-label">Clear App Cache</span>
            <p className="setting-desc">Remove all cached session data and temporary storage. Settings are preserved.</p>
          </div>
          <div className="setting-control">
            <button
              id="clear-cache-btn"
              type="button"
              className="btn-secondary"
              style={{ width: "100%" }}
              onClick={handleClearCache}
            >
              {cacheClearDone ? (
                <><CheckCircle size={16} style={{ color: "var(--accent-success)" }} /> Cleared</>
              ) : (
                <><Trash2 size={16} /> Clear Cache</>
              )}
            </button>
          </div>
        </div>

        {/* App version info */}
        <div className="setting-row" style={{ borderBottom: "none" }}>
          <div className="setting-label-wrap">
            <span className="setting-label">Application Version</span>
            <p className="setting-desc">Eyeva AI Visual Assistant — built for visually impaired users.</p>
          </div>
          <div className="setting-control">
            <div
              style={{
                padding: "10px 14px",
                background: "var(--bg-secondary)",
                border: "1px solid var(--border-default)",
                borderRadius: 8,
                fontSize: 12,
                fontFamily: "var(--font-mono)",
                color: "var(--text-muted)",
              }}
            >
              v2.1.0 · Production
            </div>
          </div>
        </div>
      </Section>

      {/* ── Save / Reset Action Bar ───────────────────────────────────────── */}
      <div
        style={{
          display: "flex",
          gap: 12,
          paddingTop: 8,
          position: "sticky",
          bottom: 16,
          background: "var(--bg-primary)",
          padding: "16px 0",
          borderTop: "1px solid var(--border-subtle)",
          marginTop: 8,
          zIndex: 10,
        }}
        role="group"
        aria-label="Settings actions"
      >
        <button
          id="save-settings-btn"
          type="button"
          onClick={handleSave}
          className="btn-primary"
          style={{ flex: 1 }}
        >
          {saved ? (
            <><CheckCircle size={17} aria-hidden="true" /> Saved Successfully!</>
          ) : (
            <><Zap size={17} aria-hidden="true" /> Save Configuration</>
          )}
        </button>

        <button
          id="reset-settings-btn"
          type="button"
          onClick={() => setShowResetModal(true)}
          className="btn-ghost"
          style={{ gap: 8, padding: "0 20px" }}
        >
          <RotateCcw size={16} aria-hidden="true" />
          Reset
        </button>
      </div>

      {/* ── Reset Confirmation Modal ──────────────────────────────────────── */}
      {showResetModal && (
        <ConfirmModal
          title="Reset All Settings?"
          message="This will restore all configuration to factory defaults. Your settings will be permanently lost and cannot be recovered."
          confirmLabel="Reset to Defaults"
          onConfirm={handleReset}
          onCancel={() => setShowResetModal(false)}
        />
      )}
    </div>
  );
}
