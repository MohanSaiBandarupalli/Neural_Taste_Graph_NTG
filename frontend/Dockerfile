FROM node:22-slim AS deps
WORKDIR /web
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

FROM node:22-slim AS build
WORKDIR /web
# Next.js resolves rewrites() at BUILD time, so the API origin is a build argument.
ARG NTG_API_URL=http://api:8000
ENV NEXT_TELEMETRY_DISABLED=1 NTG_API_URL=$NTG_API_URL
COPY --from=deps /web/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:22-slim AS runtime
WORKDIR /web
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1
COPY --from=build /web/.next/standalone ./
COPY --from=build /web/.next/static ./.next/static
USER node
EXPOSE 3000
CMD ["node", "server.js"]
