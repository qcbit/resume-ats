# --- Build Stage (React App) ---
FROM node:23-alpine AS build

WORKDIR /app

# Copy package files first for better caching
COPY package*.json ./
RUN npm install

COPY . .

# Build the app
RUN npm run build

# --- Final Stage (Caddy Server) ---
FROM caddy:2-alpine

# Remove default Caddyfile from the image
RUN rm -f /etc/caddy/Caddyfile

# Copy your custom Caddyfile
COPY Caddyfile /etc/caddy/Caddyfile

# Copy built React app assets from the build stage
COPY --from=build /app/dist /usr/share/caddy

# Caddy automatically exposes port 80.
# The default CMD in the caddy image runs Caddy with the config.