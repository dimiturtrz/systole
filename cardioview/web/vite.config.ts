import { defineConfig } from 'vite';

// Static SPA. glb + manifest are served from public/data/.
export default defineConfig({
  base: './',
  build: { target: 'es2020', chunkSizeWarningLimit: 4000 },
});
