/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Lint is run as a separate step (npm run lint); don't fail the production
  // `next build` on lint findings. Type errors are NOT ignored — they still
  // fail the build, which is what we want.
  eslint: {
    ignoreDuringBuilds: true,
  },
  // nginx fronts /api and /ws, so no rewrites are needed here.
  images: {
    // Heatmaps and uploads are rendered from base64 data URLs, so no remote
    // image optimization is required. Keeping this explicit and minimal.
    remotePatterns: [],
  },
};

export default nextConfig;
