# Argus Vision frontend container.
# Single-stage build for simplicity. Next.js inlines NEXT_PUBLIC_* at build time,
# so the public URLs must be present as ENV before `npm run build`.
FROM node:20-alpine

WORKDIR /app

# Build-time arguments promoted to ENV so Next inlines them into the bundle.
ARG NODE_ENV=production
ARG NEXT_PUBLIC_API_URL=http://localhost/api
ARG NEXT_PUBLIC_WS_URL=ws://localhost/ws
ENV NODE_ENV=$NODE_ENV
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NEXT_PUBLIC_WS_URL=$NEXT_PUBLIC_WS_URL

# Install dependencies first to leverage Docker layer caching.
# The package-lock.json is optional; the glob keeps the COPY valid when absent.
# NODE_ENV=production is set above, so devDependencies (typescript, tailwindcss,
# postcss, autoprefixer, eslint, @types/*) would be omitted by default. They are
# required by `next build`, so force their installation with --include=dev.
COPY package.json package-lock.json* ./
RUN npm install --include=dev

# Copy the rest of the application source and build the production bundle.
COPY . .
RUN npm run build

EXPOSE 3000

# Default to the production server.
# For local development, override the command, e.g.:
#   docker run ... npm run dev
CMD ["npm", "start"]
