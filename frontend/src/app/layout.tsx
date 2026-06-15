import type { Metadata } from "next";
import "./globals.css";
import Navigation from "@/components/Navigation";

export const metadata: Metadata = {
  title: "Eyeva AI — Visual Assistant for the Visually Impaired",
  description:
    "AI-powered visual assistant that describes scenes, reads text, and answers questions through voice interaction — built for visually impaired users.",
  keywords: ["visual assistant", "AI", "accessibility", "visually impaired", "voice", "OCR"],
  robots: "index, follow",
  openGraph: {
    title: "Eyeva AI V1",
    description: "See the world through AI — voice-controlled visual assistant.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <head>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <meta name="theme-color" content="#07071a" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  var s = localStorage.getItem('eyeva-settings');
                  if (s) {
                    var parsed = JSON.parse(s);
                    if (parsed.textSize && parsed.textSize !== 'normal') {
                      document.documentElement.classList.add('text-size-' + parsed.textSize);
                    }
                    if (parsed.contrastMode === 'high') {
                      document.documentElement.classList.add('contrast-high');
                    }
                    if (parsed.colorTheme && parsed.colorTheme !== 'default') {
                      document.documentElement.classList.add('theme-' + parsed.colorTheme);
                    }
                  }
                } catch (e) {}
              })();
            `
          }}
        />
      </head>
      <body className="bg-eyeva min-h-screen">
        {/* Skip to content — accessibility */}
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:fixed focus:top-4 focus:left-4 focus:z-50 btn-primary"
        >
          Skip to main content
        </a>

        {/* Ambient background orbs */}
        <div className="fixed inset-0 pointer-events-none overflow-hidden" aria-hidden="true">
          <div
            className="absolute rounded-full blur-3xl opacity-10"
            style={{
              width: "600px",
              height: "600px",
              top: "-200px",
              left: "-200px",
              background: "radial-gradient(circle, #00d4ff, transparent 70%)",
            }}
          />
          <div
            className="absolute rounded-full blur-3xl opacity-8"
            style={{
              width: "500px",
              height: "500px",
              bottom: "-150px",
              right: "-100px",
              background: "radial-gradient(circle, #a855f7, transparent 70%)",
            }}
          />
        </div>

        {/* Navigation */}
        <Navigation />

        {/* Main content */}
        <main
          id="main-content"
          className="relative z-10 pt-20 min-h-screen"
          tabIndex={-1}
        >
          {children}
        </main>
      </body>
    </html>
  );
}
