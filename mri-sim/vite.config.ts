import { defineConfig } from 'vite';

// base matters for GitHub Pages later (e.g. '/mri-sim/'); default for local dev.
// optimizeDeps sourcemap off: vtk.js ships degenerate .js.map files → Firefox logs
// "No sources are declared in this source map" for each dep. Dropping the sourcemap
// link on pre-bundled deps silences that dev-console noise.
export default defineConfig({
  base: './',
  optimizeDeps: {
    esbuildOptions: { sourcemap: false },
  },
});
