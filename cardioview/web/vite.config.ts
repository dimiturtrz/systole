import { defineConfig } from 'vite';

// Static SPA. glb + manifest are served from public/data/.
export default defineConfig({
  base: './',
  build: { target: 'es2022', chunkSizeWarningLimit: 4000 }, // top-level await in main.ts
});
