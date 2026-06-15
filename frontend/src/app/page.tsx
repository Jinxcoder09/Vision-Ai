"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import {
  Camera,
  Mic,
  BookOpen,
  Settings,
  Eye,
  Zap,
  Activity,
  ChevronRight,
  Wifi,
  WifiOff,
  Compass,
  Coins,
} from "lucide-react";
import { checkHealth, type HealthResponse } from "@/lib/api";

// ── Feature Card ──────────────────────────────────────────────────────────────

interface FeatureCardProps {
  href: string;
  icon: React.ReactNode;
  title: string;
  description: string;
  shortcut: string;
  accentColor: string;
  id: string;
}

function FeatureCard({
  href,
  icon,
  title,
  description,
  shortcut,
  accentColor,
  id,
}: FeatureCardProps) {
  return (
    <Link
      href={href}
      id={id}
      className="glass-card p-6 flex flex-col gap-4 cursor-pointer group focus:outline-none focus:ring-4 focus:ring-cyan-400"
      aria-label={`${title} — ${description}. Press key ${shortcut} to open`}
      style={{ minHeight: "220px" }}
    >
      {/* Icon */}
      <div
        className="w-16 h-16 rounded-2xl flex items-center justify-center transition-transform group-hover:scale-105"
        style={{ background: `${accentColor}18`, border: `2px solid ${accentColor}40` }}
        aria-hidden="true"
      >
        <div style={{ color: accentColor }}>{icon}</div>
      </div>

      {/* Content */}
      <div className="flex-1">
        <h2 className="text-2xl font-black text-white mb-2 tracking-tight flex items-center gap-2">
          {title}
        </h2>
        <p className="text-slate-300 text-sm leading-relaxed font-medium">{description}</p>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between mt-2">
        <kbd
          className="px-3 py-1 rounded-md text-xs font-black font-mono"
          style={{
            background: `${accentColor}15`,
            color: accentColor,
            border: `2px solid ${accentColor}35`,
          }}
          aria-label={`Keyboard shortcut key ${shortcut}`}
        >
          Key {shortcut}
        </kbd>
        <ChevronRight
          size={22}
          className="text-slate-500 group-hover:text-cyan-400 group-hover:translate-x-1 transition-all"
          aria-hidden="true"
        />
      </div>
    </Link>
  );
}

// ── Status Badge ─────────────────────────────────────────────────────────────

function ServiceBadge({ name, ok }: { name: string; ok: boolean }) {
  return (
    <div className="flex items-center gap-2.5 text-sm font-semibold" role="status" aria-label={`${name}: ${ok ? "online" : "offline"}`}>
      <div
        className={`w-3.5 h-3.5 rounded-full ${ok ? "bg-emerald-400" : "bg-red-500"}`}
        style={{ boxShadow: ok ? "0 0 8px rgba(52,211,153,0.7)" : "0 0 8px rgba(239,68,68,0.7)" }}
        aria-hidden="true"
      />
      <span className={ok ? "text-slate-200" : "text-slate-400"}>{name}</span>
    </div>
  );
}

// ── Hero Wave ─────────────────────────────────────────────────────────────────

function HeroWave() {
  return (
    <div className="flex items-end gap-1.5 h-10" aria-hidden="true">
      {[28, 40, 22, 36, 18, 32, 26].map((h, i) => (
        <div
          key={i}
          className="wave-bar rounded-full"
          style={{
            height: `${h}px`,
            animationDelay: `${i * 0.12}s`,
            background: i % 2 === 0
              ? "var(--accent-cyan)"
              : "var(--accent-violet)",
          }}
        />
      ))}
    </div>
  );
}

// ── Dashboard Page ────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);

  const fetchHealth = useCallback(async () => {
    try {
      const h = await checkHealth();
      setHealth(h);
    } catch {
      setHealth(null);
    } finally {
      setHealthLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 30_000);
    return () => clearInterval(interval);
  }, [fetchHealth]);

  const features: FeatureCardProps[] = [
    {
      href: "/camera?auto=true",
      id: "card-start-assistant",
      icon: <Mic size={30} strokeWidth={2.5} />,
      title: "Start Assistant",
      description:
        "Initialize visual guide and continuous voice assistant immediately (hands-free).",
      shortcut: "1",
      accentColor: "#00d4ff",
    },
    {
      href: "/camera",
      id: "card-scene-understanding",
      icon: <Camera size={30} strokeWidth={2.5} />,
      title: "Scene Understanding",
      description:
        "Analyze your general surroundings — identify items, people, layout, and obstacles.",
      shortcut: "2",
      accentColor: "#a855f7",
    },
    {
      href: "/reader?auto=true",
      id: "card-text-reader",
      icon: <BookOpen size={30} strokeWidth={2.5} />,
      title: "Text Reader",
      description:
        "Continuous OCR reader. Point camera at signs, packages, or books to read aloud.",
      shortcut: "3",
      accentColor: "#10b981",
    },
    {
      href: "/navigation?auto=true",
      id: "card-navigation-mode",
      icon: <Compass size={30} strokeWidth={2.5} />,
      title: "Navigation Mode",
      description:
        "AR path visualizer with collision avoidance and real-time path alerts.",
      shortcut: "4",
      accentColor: "#3b82f6",
    },
    {
      href: "/money?auto=true",
      id: "card-money-recognition",
      icon: <Coins size={30} strokeWidth={2.5} />,
      title: "Money Recognition",
      description:
        "Count bank notes, identify currency values, and calculate running cash totals.",
      shortcut: "5",
      accentColor: "#eab308",
    },
    {
      href: "/settings",
      id: "card-settings",
      icon: <Settings size={30} strokeWidth={2.5} />,
      title: "Settings",
      description:
        "Manage voice pitch/volume, mic sensitivity, custom camera bounds, and accessibility themes.",
      shortcut: "6",
      accentColor: "#f97316",
    },
  ];

  // Keyboard numeric shortcuts 1-6
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
      switch (e.key) {
        case "1": window.location.href = "/camera?auto=true"; break;
        case "2": window.location.href = "/camera"; break;
        case "3": window.location.href = "/reader?auto=true"; break;
        case "4": window.location.href = "/navigation?auto=true"; break;
        case "5": window.location.href = "/money?auto=true"; break;
        case "6": window.location.href = "/settings"; break;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="max-w-6xl mx-auto px-6 py-12">
      {/* ── Hero ─────────────────────────────────────────────────────────── */}
      <section className="mb-14" aria-labelledby="hero-heading">
        <div className="flex items-center gap-4 mb-6">
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center"
            style={{
              background: "linear-gradient(135deg, #00d4ff22, #a855f722)",
              border: "2px solid rgba(0,212,255,0.3)",
            }}
            aria-hidden="true"
          >
            <Eye size={32} style={{ color: "#00d4ff" }} />
          </div>
          <HeroWave />
        </div>

        <h1
          id="hero-heading"
          className="text-5xl sm:text-6xl font-extrabold mb-4 leading-tight tracking-tight text-white"
        >
          <span className="gradient-text">See the world</span>
          <br />
          through AI.
        </h1>
        <p className="text-slate-300 text-lg sm:text-xl max-w-2xl leading-relaxed mb-8 font-medium">
          Eyeva is a voice-first visual companion for visually impaired users. 
          Use hotkeys 1-6 or voice activation to navigate the features.
        </p>

        <div className="flex flex-wrap gap-4">
          <Link
            href="/camera?auto=true"
            id="hero-start-btn"
            className="btn-primary px-8 py-4 text-base font-bold shadow-lg"
            aria-label="Start Voice Assistant immediately"
          >
            <Zap size={20} aria-hidden="true" />
            Start Assistant
          </Link>
          <Link
            href="/navigation?auto=true"
            id="hero-nav-btn"
            className="btn-secondary px-8 py-4 text-base font-bold"
            aria-label="Open Navigation Mode"
          >
            <Compass size={20} aria-hidden="true" />
            Navigation
          </Link>
        </div>
      </section>

      {/* ── Feature Portals Grid ─────────────────────────────────────────── */}
      <section aria-labelledby="portals-heading" className="mb-14">
        <h2
          id="portals-heading"
          className="text-sm font-black text-slate-400 mb-6 uppercase tracking-widest"
        >
          Assistive Portals
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((f) => (
            <FeatureCard key={f.href + f.id} {...f} />
          ))}
        </div>
      </section>

      {/* ── System Status Telemetry ─────────────────────────────────────────── */}
      <section
        className="glass-card p-6"
        aria-labelledby="status-heading"
        role="region"
      >
        <div className="flex items-center justify-between mb-5 border-b border-cyan-500/10 pb-4">
          <h2 id="status-heading" className="font-bold text-slate-200 text-lg flex items-center gap-2">
            <Activity className="w-5 h-5 text-cyan-400" aria-hidden="true" />
            System Status
          </h2>
          <div className="flex items-center gap-2 text-sm font-semibold animate-pulse" aria-live="polite" aria-atomic="true">
            {healthLoading ? (
              <span className="text-slate-400">Verifying...</span>
            ) : health ? (
              <>
                <Wifi size={16} className="text-emerald-400" aria-hidden="true" />
                <span className={health.status === "ok" ? "text-emerald-400" : "text-amber-400"}>
                  {health.status === "ok" ? "All services online" : "Degraded Link"}
                </span>
              </>
            ) : (
              <>
                <WifiOff size={16} className="text-red-500" aria-hidden="true" />
                <span className="text-red-500">Backend offline</span>
              </>
            )}
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          {health ? (
            Object.entries(health.services).map(([name, ok]) => (
              <ServiceBadge
                key={name}
                name={name.toUpperCase()}
                ok={ok as boolean}
              />
            ))
          ) : (
            ["STT", "TTS", "OCR", "VISION"].map((name) => (
              <ServiceBadge key={name} name={name} ok={false} />
            ))
          )}
        </div>

        {!health && !healthLoading && (
          <p className="mt-4 text-sm text-red-400 font-semibold" role="alert">
            ⚠ Connection to the local backend failed. Verify the server is running on port 8000.
          </p>
        )}
      </section>
    </div>
  );
}
