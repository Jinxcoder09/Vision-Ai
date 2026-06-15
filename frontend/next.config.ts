import type { NextConfig } from "next";

const isProd = process.env.NODE_ENV === "production";

const nextConfig: NextConfig = {
  // Build Next.js as a static HTML export in production
  output: isProd ? "export" : undefined,
  
  // Disable next-dev API proxying/rewrites in static build
  ...(isProd ? {} : {
    async rewrites() {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";
      return [
        {
          source: "/api/:path*",
          destination: `${apiUrl}/api/:path*`,
        },
      ];
    },
  }),
  
  images: {
    unoptimized: true, // required for static HTML export
  },
  
  reactStrictMode: true,
};

export default nextConfig;
