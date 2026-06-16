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
    title: "Eyeva AI",
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
        <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
        <meta name="theme-color" content="#0A0E27" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        {/* Accessibility: inject theme classes before first paint to prevent flash */}
        <script
          dangerouslySetInnerHTML={{
            __html: `
              (function() {
                try {
                  var s = localStorage.getItem('eyeva-settings');
                  if (s) {
                    var p = JSON.parse(s);
                    if (p.textSize && p.textSize !== 'normal') {
                      document.documentElement.classList.add('text-size-' + p.textSize);
                    }
                    if (p.contrastMode === 'high') {
                      document.documentElement.classList.add('contrast-high');
                    }
                    if (p.colorTheme && p.colorTheme !== 'default') {
                      document.documentElement.classList.add('theme-' + p.colorTheme);
                    }
                  }
                } catch(e) {}
              })();
            `,
          }}
        />
      </head>
      <body className="bg-eyeva">
        {/* Skip to main content — accessibility */}
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:fixed focus:top-4 focus:left-4 focus:z-50 btn-primary"
        >
          Skip to main content
        </a>

        {/* Ambient background orbs */}
        <div className="fixed inset-0 pointer-events-none overflow-hidden" aria-hidden="true">
          <div
            style={{
              position: "absolute",
              width: 600,
              height: 600,
              top: -200,
              left: -200,
              borderRadius: "50%",
              background: "radial-gradient(circle, rgba(0,153,204,0.08), transparent 70%)",
              filter: "blur(60px)",
            }}
          />
          <div
            style={{
              position: "absolute",
              width: 500,
              height: 500,
              bottom: -150,
              right: -100,
              borderRadius: "50%",
              background: "radial-gradient(circle, rgba(106,90,205,0.06), transparent 70%)",
              filter: "blur(60px)",
            }}
          />
        </div>

        {/* Navigation — sidebar (desktop/tablet) + top bar + bottom tabs (mobile) */}
        <Navigation />

        {/* Main content area */}
        <main
          id="main-content"
          className="main-content relative z-10"
          tabIndex={-1}
        >
          {children}
        </main>
      </body>
    </html>
  );
}
