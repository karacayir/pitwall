import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // Fully client-rendered app: static export deploys anywhere (Cloudflare Pages).
  output: "export",
};

export default nextConfig;
