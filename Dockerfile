FROM node:22-alpine AS base

# Stage 1: install deps
FROM base AS deps
WORKDIR /usr/src/app
COPY package*.json ./
RUN npm ci

# Stage 2: build the app
FROM base AS builder
WORKDIR /usr/src/app
COPY --from=deps /usr/src/app/node_modules ./node_modules
COPY . .
RUN npm run build

# Stage 3: server
FROM nginx:1.27.3-alpine-slim
COPY --from=builder /usr/src/app/build /usr/share/nginx/html