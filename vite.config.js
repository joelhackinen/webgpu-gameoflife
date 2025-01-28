import { defineConfig } from "vite";

export default defineConfig({
  build: {
    target: "es2022", // or 'esnext' for the most modern features
    outDir: "build",
  },
});
