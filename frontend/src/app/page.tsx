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
  LayoutDashboard,
  ArrowRight,
} from "lucide-react";
import { checkHealth, type HealthResponse } from "@/lib/api";

// ── Feature Card ─────────────────────────────────────────────────────────────

interface FeatureCardProps {
  href: string;
  icon: React.ReactNode;
  title: string;
  description: string;
  shortcut: string;
  accentColor: string;
  id: string;
  delay?: number;
}

function FeatureCard({ href, icon, title, description, shortcut, accentColor, id, delay = 0 }: FeatureCardProps) {
  return (
    <Link
      href={href}
      id={id}
      className="feature-card group focus:outline-none"
      aria-label={`${title} — ${description}. Press key ${shortcut} to open`}
      style={{
        borderLeftColor: accentColor,
        animationDelay: `${delay}ms`,
      }}
    >
      {/* Top row: icon + shortcut */}
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between" }}>
        <div
          style={{
            width: 44,
            height: 44,
            borderRadius: 10,
            background: `${accentColor}18`,
            border: `1.5px solid ${accentColor}40`,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: accentColor,
            flexShrink: 0,
            transition: "transform 0.2s ease",
          }}
          className="group-hover:scale-110"
          aria-hidden="true"
        >
          {icon}
        </div>
        <kbd
          style={{
            padding: "3px 8px",
            borderRadius: 6,
            fontSize: 11,
            fontWeight: 700,
            fontFamily: "var(--font-mono)",
            background: `${accentColor}12`,
            color: accentColor,
            border: `1px solid ${accentColor}30`,
          }}
          aria-label={`Keyboard shortcut ${shortcut}`}
        >
          {shortcut}
        </kbd>
      </div>

      {/* Content */}
      <div style={{ flex: 1 }}>
        <h2 style={{ fontSize: 16, fontWeight: 700, color: "var(--text-primary)", marginBottom: 5 }}>
          {title}
        </h2>
        <p style={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.5 }}>
          {description}
        </p>
      </div>

      {/* Footer arrow */}
      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <ChevronRight
          size={16}
          style={{ color: "var(--text-muted)", transition: "all 0.2s ease" }}
          className="group-hover:text-[var(--accent-primary)] group-hover:translate-x-1"
          aria-hidden="true"
        />
      </div>
    </Link>
  );
}

// ── Service Status Badge ──────────────────────────────────────────────────────

function ServiceBadge({ name, ok }: { name: string; ok: boolean }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "10px 14px",
        background: ok ? "rgba(0,200,83,0.06)" : "rgba(255,82,82,0.06)",
        border: `1px solid ${ok ? "rgba(0,200,83,0.2)" : "rgba(255,82,82,0.2)"}`,
        borderRadius: 8,
      }}
      role="status"
      aria-label={`${name}: ${ok ? "online" : "offline"}`}
    >
      <div
        className={`status-dot ${ok ? "online" : "error"}`}
        aria-hidden="true"
      />
      <span style={{ fontSize: 12, fontWeight: 600, color: ok ? "var(--accent-success)" : "var(--accent-error)" }}>
        {name}
      </span>
    </div>
  );
}

// ── Waveform ──────────────────────────────────────────────────────────────────

function HeroWave() {
  return (
    <div style={{ display: "flex", alignItems: "flex-end", gap: 3, height: 32 }} aria-hidden="true">
      {[20, 32, 16, 28, 12, 24, 18].map((h, i) => (
        <div
          key={i}
          className="wave-bar"
          style={{ height: h, animationDelay: `${i * 0.1}s` }}
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
      icon: <Mic size={22} strokeWidth={2.5} />,
      title: "Start Assistant",
      description: "Initialize visual guide and continuous voice assistant immediately (hands-free).",
      shortcut: "1",
      accentColor: "#0099CC",
      delay: 0,
    },
    {
      href: "/camera",
      id: "card-scene-understanding",
      icon: <Camera size={22} strokeWidth={2.5} />,
      title: "Scene Understanding",
      description: "Analyze surroundings — identify objects, people, layout, and hazards.",
      shortcut: "2",
      accentColor: "#6A5ACD",
      delay: 60,
    },
    {
      href: "/reader?auto=true",
      id: "card-text-reader",
      icon: <BookOpen size={22} strokeWidth={2.5} />,
      title: "Text Reader",
      description: "Point camera at signs, packages, or books to read aloud via OCR.",
      shortcut: "3",
      accentColor: "#00C853",
      delay: 120,
    },
    {
      href: "/navigation?auto=true",
      id: "card-navigation-mode",
      icon: <Compass size={22} strokeWidth={2.5} />,
      title: "Navigation Mode",
      description: "AR path visualizer with collision avoidance and real-time path alerts.",
      shortcut: "4",
      accentColor: "#0099CC",
      delay: 180,
    },
    {
      href: "/money?auto=true",
      id: "card-money-recognition",
      icon: <Coins size={22} strokeWidth={2.5} />,
      title: "Money Recognition",
      description: "Count banknotes, identify currency values, and calculate running totals.",
      shortcut: "5",
      accentColor: "#FF9800",
      delay: 240,
    },
    {
      href: "/settings",
      id: "card-settings",
      icon: <Settings size={22} strokeWidth={2.5} />,
      title: "Settings",
      description: "Manage voice, microphone, camera, accessibility, and AI preferences.",
      shortcut: "6",
      accentColor: "#6A5ACD",
      delay: 300,
    },
  ];

  // Keyboard shortcuts 1–6
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement || e.target instanceof HTMLTextAreaElement) return;
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
    <div style={{ maxWidth: 1100, margin: "0 auto", padding: "40px 24px 32px" }}>

      {/* ── Hero Section ─────────────────────────────────────────────────── */}
      <section aria-labelledby="hero-heading" style={{ marginBottom: 48 }}>

        {/* Logo + wave */}
        <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 24 }}>
          <div
            style={{
              width: 52,
              height: 52,
              borderRadius: 14,
              background: "rgba(0,153,204,0.12)",
              border: "1.5px solid rgba(0,153,204,0.3)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
            aria-hidden="true"
          >
            <Eye size={26} style={{ color: "var(--accent-primary)" }} />
          </div>
          <HeroWave />
        </div>

        <h1
          id="hero-heading"
          style={{
            fontSize: "clamp(32px, 6vw, 52px)",
            fontWeight: 800,
            lineHeight: 1.15,
            letterSpacing: "-0.02em",
            marginBottom: 14,
            color: "var(--text-primary)",
          }}
        >
          <span className="gradient-text">See the world</span>
          <br />
          through AI.
        </h1>

        <p
          style={{
            fontSize: "clamp(15px, 2vw, 18px)",
            color: "var(--text-secondary)",
            maxWidth: 520,
            lineHeight: 1.6,
            marginBottom: 28,
          }}
        >
          Eyeva is a voice-first visual companion for visually impaired users.
          Use hotkeys <kbd style={{ fontFamily: "var(--font-mono)", background: "rgba(255,255,255,0.08)", padding: "1px 6px", borderRadius: 4, fontSize: 13 }}>1</kbd>–<kbd style={{ fontFamily: "var(--font-mono)", background: "rgba(255,255,255,0.08)", padding: "1px 6px", borderRadius: 4, fontSize: 13 }}>6</kbd> or voice activation to navigate.
        </p>

        <div style={{ display: "flex", flexWrap: "wrap", gap: 12 }}>
          <Link
            href="/camera?auto=true"
            id="hero-start-btn"
            className="btn-primary"
            style={{ fontSize: 15, height: 48, padding: "0 28px" }}
            aria-label="Start Voice Assistant immediately"
          >
            <Zap size={18} aria-hidden="true" />
            Start Assistant
          </Link>
          <Link
            href="/navigation?auto=true"
            id="hero-nav-btn"
            className="btn-secondary"
            style={{ fontSize: 15, height: 48, padding: "0 28px" }}
            aria-label="Open Navigation Mode"
          >
            <Compass size={18} aria-hidden="true" />
            Navigation
          </Link>
        </div>
      </section>

      {/* ── Feature Portals Grid ─────────────────────────────────────────── */}
      <section aria-labelledby="portals-heading" style={{ marginBottom: 40 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
          <h2
            id="portals-heading"
            style={{
              fontSize: 11,
              fontWeight: 700,
              color: "var(--text-muted)",
              textTransform: "uppercase",
              letterSpacing: "0.12em",
            }}
          >
            <LayoutDashboard size={12} style={{ display: "inline", marginRight: 6, verticalAlign: "middle" }} />
            Assistive Portals
          </h2>
          <span style={{ fontSize: 11, color: "var(--text-muted)" }}>
            Press 1–6 to jump instantly
          </span>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
            gap: 16,
          }}
        >
          {features.map((f) => (
            <FeatureCard key={f.id} {...f} />
          ))}
        </div>
      </section>

      {/* ── System Status ────────────────────────────────────────────────── */}
      <section
        className="glass-card"
        style={{ padding: "20px 24px" }}
        aria-labelledby="status-heading"
        role="region"
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: 16,
            paddingBottom: 14,
            borderBottom: "1px solid var(--border-subtle)",
            flexWrap: "wrap",
            gap: 10,
          }}
        >
          <h2
            id="status-heading"
            style={{
              fontWeight: 700,
              fontSize: 14,
              color: "var(--text-primary)",
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            <Activity size={16} style={{ color: "var(--accent-primary)" }} aria-hidden="true" />
            System Status
          </h2>

          <div
            style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12, fontWeight: 600 }}
            aria-live="polite"
            aria-atomic="true"
          >
            {healthLoading ? (
              <><div className="spinner" style={{ width: 14, height: 14, borderWidth: 2 }} /><span style={{ color: "var(--text-muted)" }}>Checking...</span></>
            ) : health ? (
              <>
                <Wifi size={14} style={{ color: health.status === "ok" ? "var(--accent-success)" : "var(--accent-warning)" }} aria-hidden="true" />
                <span style={{ color: health.status === "ok" ? "var(--accent-success)" : "var(--accent-warning)" }}>
                  {health.status === "ok" ? "All systems online" : "Degraded service"}
                </span>
              </>
            ) : (
              <>
                <WifiOff size={14} style={{ color: "var(--accent-error)" }} aria-hidden="true" />
                <span style={{ color: "var(--accent-error)" }}>Backend offline</span>
              </>
            )}
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))",
            gap: 8,
          }}
        >
          {health ? (
            Object.entries(health.services).map(([name, ok]) => (
              <ServiceBadge key={name} name={name.toUpperCase()} ok={ok as boolean} />
            ))
          ) : (
            ["STT", "TTS", "OCR", "VISION"].map((name) => (
              <ServiceBadge key={name} name={name} ok={false} />
            ))
          )}
        </div>

        {!health && !healthLoading && (
          <p
            style={{ marginTop: 14, fontSize: 13, color: "var(--accent-error)", fontWeight: 500, display: "flex", alignItems: "center", gap: 6 }}
            role="alert"
          >
            ⚠ Cannot reach backend. Ensure the server is running on port 8000.
          </p>
        )}

        {/* Quick links */}
        <div style={{ display: "flex", gap: 8, marginTop: 16, paddingTop: 14, borderTop: "1px solid var(--border-subtle)", flexWrap: "wrap" }}>
          {[
            { href: "/camera", label: "Open Vision" },
            { href: "/reader", label: "Open Reader" },
            { href: "/settings", label: "Configure" },
          ].map(({ href, label }) => (
            <Link
              key={href}
              href={href}
              className="btn-ghost"
              style={{ fontSize: 12, height: 32, padding: "0 12px" }}
            >
              {label}
              <ArrowRight size={12} aria-hidden="true" />
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
