"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Eye, Camera, BookOpen, Settings, Compass, Coins } from "lucide-react";

const navItems = [
  { href: "/", label: "Dashboard", icon: Eye, id: "nav-dashboard" },
  { href: "/camera", label: "AI Vision", icon: Camera, id: "nav-camera" },
  { href: "/reader", label: "Text Reader", icon: BookOpen, id: "nav-reader" },
  { href: "/navigation", label: "Navigation", icon: Compass, id: "nav-navigation" },
  { href: "/money", label: "Money", icon: Coins, id: "nav-money" },
  { href: "/settings", label: "Settings", icon: Settings, id: "nav-settings" },
];

export default function Navigation() {
  const pathname = usePathname();

  return (
    <nav
      className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4"
      style={{
        background: "rgba(7, 7, 26, 0.85)",
        backdropFilter: "blur(20px)",
        borderBottom: "1px solid rgba(255,255,255,0.06)",
      }}
      role="navigation"
      aria-label="Main navigation"
    >
      {/* Logo */}
      <Link
        href="/"
        className="flex items-center gap-3 group"
        id="nav-logo"
        aria-label="Eyeva AI — Go to dashboard"
      >
        <div
          className="w-9 h-9 rounded-xl flex items-center justify-center"
          style={{
            background: "linear-gradient(135deg, #00d4ff, #a855f7)",
          }}
          aria-hidden="true"
        >
          <Eye size={18} color="#000" strokeWidth={2.5} />
        </div>
        <span className="font-bold text-lg tracking-tight">
          <span className="gradient-text">Eyeva</span>
          <span className="text-white/40 font-light ml-1 text-sm">AI</span>
        </span>
      </Link>

      {/* Nav links */}
      <div className="hidden md:flex items-center gap-1" role="list">
        {navItems.map(({ href, label, icon: Icon, id }) => {
          const isActive = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              id={id}
              role="listitem"
              className={`nav-link ${isActive ? "active" : ""}`}
              aria-current={isActive ? "page" : undefined}
            >
              <Icon size={16} aria-hidden="true" />
              {label}
            </Link>
          );
        })}
      </div>

      {/* Mobile nav — icon only */}
      <div className="flex md:hidden items-center gap-2" role="list">
        {navItems.map(({ href, label, icon: Icon, id }) => {
          const isActive = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              id={`${id}-mobile`}
              className={`p-2.5 rounded-xl transition-all ${
                isActive
                  ? "bg-cyan-500/10 text-cyan-400"
                  : "text-slate-500 hover:text-slate-300"
              }`}
              aria-label={label}
              aria-current={isActive ? "page" : undefined}
            >
              <Icon size={20} aria-hidden="true" />
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
