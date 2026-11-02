/** Proxies /api/* to the FastAPI backend so the browser talks to one origin (no CORS, no exposed backend URL). */
const API = process.env.NTG_API_URL || "http://127.0.0.1:8000";
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/:path*` }];
  },
  async headers() {
    return [{ source: "/:path*", headers: [
      { key: "X-Frame-Options", value: "DENY" },
      { key: "X-Content-Type-Options", value: "nosniff" },
      { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
    ] }];
  },
};
export default nextConfig;
