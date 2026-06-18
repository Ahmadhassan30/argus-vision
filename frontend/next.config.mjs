/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // nginx fronts /api and /ws, so no rewrites are needed here.
  images: {
    // Heatmaps and uploads are rendered from base64 data URLs, so no remote
    // image optimization is required. Keeping this explicit and minimal.
    remotePatterns: [],
  },
};

export default nextConfig;
