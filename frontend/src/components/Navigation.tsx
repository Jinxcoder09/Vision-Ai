"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Eye,
  Camera,
  BookOpen,
  Settings,
  Compass,
  Coins,
  LayoutDashboard,
} from "lucide-react";

const navItems = [
  { href: "/",           label: "Dashboard",  icon: LayoutDashboard, id: "nav-dashboard" },
  { href: "/camera",     label: "AI Vision",  icon: Camera,          id: "nav-camera" },
  { href: "/reader",     label: "Text Reader",icon: BookOpen,        id: "nav-reader" },
  { href: "/navigation", label: "Navigation", icon: Compass,         id: "nav-navigation" },
  { href: "/money",      label: "Money",      icon: Coins,           id: "nav-money" },
  { href: "/settings",   label: "Settings",   icon: Settings,        id: "nav-settings" },
];

// Bottom tab items (5 most important, desktop gets all in sidebar)
const tabItems = [
  { href: "/",           label: "Home",       icon: LayoutDashboard, id: "tab-dashboard" },
  { href: "/camera",     label: "Vision",     icon: Camera,          id: "tab-camera" },
  { href: "/reader",     label: "Reader",     icon: BookOpen,        id: "tab-reader" },
  { href: "/navigation", label: "Navigate",   icon: Compass,         id: "tab-navigation" },
  { href: "/settings",   label: "Settings",   icon: Settings,        id: "tab-settings" },
];

export default function Navigation() {
  const pathname = usePathname();

  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname.startsWith(href);

  return (
    <>
      {/* ── Desktop / Tablet Sidebar ─────────────────────────────────────── */}
      <aside
        className="sidebar-nav"
        role="navigation"
        aria-label="Main navigation"
      >
        {/* Logo */}
        <Link href="/" className="sidebar-logo" id="nav-logo" aria-label="Eyeva AI — Go to dashboard">
          <div className="sidebar-logo-icon" aria-hidden="true">
            <Eye size={18} color="#fff" strokeWidth={2.5} />
          </div>
          <div className="sidebar-logo-text">
            <span className="sidebar-logo-name">Eyeva AI</span>
            <span className="sidebar-logo-sub">Visual Assistant</span>
          </div>
        </Link>

        {/* Nav links */}
        <nav className="sidebar-nav-list" role="list">
          {navItems.map(({ href, label, icon: Icon, id }) => (
            <Link
              key={href}
              href={href}
              id={id}
              role="listitem"
              className={`sidebar-nav-item ${isActive(href) ? "active" : ""}`}
              aria-current={isActive(href) ? "page" : undefined}
              aria-label={label}
            >
              <Icon size={18} className="nav-icon" aria-hidden="true" />
              <span className="sidebar-label">{label}</span>
            </Link>
          ))}
        </nav>

        {/* Footer version */}
        <div
          style={{
            padding: "12px 16px",
            borderTop: "1px solid rgba(255,255,255,0.06)",
            fontSize: "11px",
            color: "var(--text-muted)",
          }}
        >
          <span className="sidebar-label">v2.1.0 — Eyeva AI</span>
        </div>
      </aside>

      {/* ── Mobile Top Bar (logo + settings icon) ───────────────────────── */}
      <header className="top-mobile-bar" aria-label="Eyeva mobile header">
        <Link href="/" className="flex items-center gap-2" aria-label="Eyeva AI home">
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: 9,
              background: "var(--accent-gradient)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0,
            }}
            aria-hidden="true"
          >
            <Eye size={16} color="#fff" strokeWidth={2.5} />
          </div>
          <span
            style={{
              fontWeight: 700,
              fontSize: 15,
              background: "var(--accent-gradient)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              backgroundClip: "text",
            }}
          >
            Eyeva AI
          </span>
        </Link>

        <Link
          href="/settings"
          className="btn-icon"
          aria-label="Settings"
          id="mobile-settings-link"
        >
          <Settings size={20} aria-hidden="true" />
        </Link>
      </header>

      {/* ── Mobile Bottom Tab Bar ────────────────────────────────────────── */}
      <nav className="bottom-tab-bar" role="navigation" aria-label="Bottom navigation">
        {tabItems.map(({ href, label, icon: Icon, id }) => (
          <Link
            key={href}
            href={href}
            id={id}
            className={`bottom-tab-item ${isActive(href) ? "active" : ""}`}
            aria-current={isActive(href) ? "page" : undefined}
            aria-label={label}
          >
            <Icon size={20} aria-hidden="true" />
            <span>{label}</span>
          </Link>
        ))}
      </nav>
    </>
  );
}
