# mri-sim

Interactive 3D MRI pipeline visualizer (spins → pulse sequence → k-space → image).
Educational; see [SPEC.md](SPEC.md) for the full design.

**Status: M4** — end-to-end pipeline visualized: a 3D grid of protons precessing on
cones (individual-spin view) with **slice-selective** excitation (the central z-slab
glows + tips), a **speed slider**, and **k-space → image** panels that fill line-by-line
(low→high frequency) so the reconstructed image sharpens. Engine is pure TS +
unit-tested (24 tests); renderer verified by an asserting visual smoke.

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
M0 scene ✓ · M1 precession + RF tip ✓ · M1.5 T1/T2 relaxation ✓ · M2 slice-select ✓ · M3 FFT/k-space engine ✓ · M4 k-space→image ✓ · M5 acquisition synced to clock ✓ · M6 pulse-sequence diagram + TR/TE ✓
M4 k-space fills → live inverse-FFT image (honest, from the spins) · M5 controls + timeline.

*Lives in the cardiac-imaging repo for now; will split to its own repo as it grows.*
