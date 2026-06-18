# mri-sim

Interactive 3D MRI pipeline visualizer (spins → pulse sequence → k-space → image).
Educational; see [SPEC.md](SPEC.md) for the full design.

**Status: M1** — animated: 90° RF tip + free precession (spins sweep the transverse
plane). Engine is pure TS + unit-tested; renderer verified by an asserting visual smoke.

## Stack
TypeScript + [vtk.js](https://kitware.github.io/vtk-js/) + Vite. Architecture is
**model ⟂ view ⟂ presenter** (engine is pure TS, renderer swappable).

```
src/
  model/      # pure-TS simulation (SpinSystem, later Sequence/Simulator) — strictly typed
  view/       # vtk.js rendering (SpinScene) — no physics
  presenter/  # wires model → view, runs the loop
  main.ts     # entry
```

## Run
```bash
cd mri-sim
npm install
npm run dev      # open the printed localhost URL
npm run build    # typecheck + production build (dist/)
```

## Milestones
M0 scene ✓ · M1 precession + RF tip ✓ · M2 slice-select · M3 phase/freq encode ·
M4 k-space fills → live inverse-FFT image (honest, from the spins) · M5 controls + timeline.

*Lives in the cardiac-imaging repo for now; will split to its own repo as it grows.*
