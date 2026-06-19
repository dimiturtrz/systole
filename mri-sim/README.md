# mri-sim

Interactive 3D MRI pipeline visualizer — **spins → pulse sequence → k-space → image**, on one clock.

![mri-sim demo](docs/media/demo.gif)

*Full-res clip: [`docs/media/demo.mp4`](docs/media/demo.mp4). Educational; see [SPEC.md](SPEC.md) for the design and [ROADMAP.md](ROADMAP.md) for what's done / next.*

**Status: pipeline complete, end-to-end.** A 3D grid of protons precesses on cones
(individual-spin view); a **slice-selective** RF pulse tips the (tiltable, oblique)
slab; **phase encode** winds the in-slice spins into a position-dependent ramp whose
steepness steps each TR (the k-space line loop); **frequency encode / readout** fills
**k-space line-by-line**, and the inverse-FFT **image** sharpens as it fills. A live
**pulse-sequence diagram** tracks the playhead. Timing is realistic (TR/TE in ms); the
dead relaxation tail is fast-forwarded so you watch the physics, not the wait. Controls:
log-scaled speed, Larmor (→ slice height), TR, TE, slice angle. Engine is pure TS,
unit-tested (42 tests); the vtk.js renderer is verified by an asserting visual smoke.

## Stack
TypeScript + [vtk.js](https://kitware.github.io/vtk-js/) + Vite. Architecture is
**model ⟂ view ⟂ presenter** (engine is pure TS, renderer swappable).

```
src/
  model/      # pure-TS simulation (SpinSystem, Simulator, Acquisition, sequence timing, FFT) — strictly typed
  view/       # rendering: vtk.js 3D spins (SpinScene) + canvas2d panels/sequence — no physics
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

*An isolated subfolder in the systole repo (own stack: TypeScript + vtk.js).*
