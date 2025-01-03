FROM node:22.12.0-bullseye-slim AS build-stage

WORKDIR /usr/src/app

COPY . .

RUN npm ci

RUN npm run build


FROM nginx:1.27.3-alpine-slim

COPY --from=build-stage /usr/src/app/build /usr/share/nginx/html