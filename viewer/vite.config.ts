import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

import { autoresearchSource } from "./server/projectSource";

const viewerRoot = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(
  process.env.AUTORESEARCH_PROJECT_ROOT ?? path.join(viewerRoot, ".."),
);
const projectAssets = path.join(projectRoot, "assets");

export default defineConfig({
  plugins: [react(), autoresearchSource(projectRoot)],
  publicDir: existsSync(projectAssets) ? projectAssets : false,
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes("@xyflow") || id.includes("@dagrejs")) return "graph";
          if (id.includes("react-markdown") || id.includes("remark-") || id.includes("unified")) {
            return "markdown";
          }
          if (id.includes("node_modules/react") || id.includes("node_modules/scheduler")) {
            return "react";
          }
          return undefined;
        },
      },
    },
  },
  server: {
    host: "127.0.0.1",
    port: 4173,
    strictPort: true,
  },
});
