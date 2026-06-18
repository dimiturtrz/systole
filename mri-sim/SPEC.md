# mri-sim — interactive 3D MRI pipeline visualizer

*Lives as an **isolated subfolder** inside the systole repo (own stack, own build).
Self-contained TypeScript app — keep it out of the Python `src/`.*

## One-liner
An interactive 3D web visualization of how an MRI image is made: the magnet + protons,
the pulse sequence (slice-select → phase-encode → frequency-encode, **repeated each TR**),
k-space filling line-by-line, and the live inverse-FFT → image.

## Why
- **Best test of understanding** — you can't fake a working simulator.
- **Portfolio piece** — an explorable "spins → k-space → image" explainer is rare and
  impressive (most ML people can't build it). Doubles as the demonstrable proof of the
  MRI-physics learning.
- It feels cool.

## Stack (leverages existing skills)
- **TypeScript** + **vtk.js** (3D scene, glyphs, image actors) — primary.
- **Vite** (dev/build), deploy to **GitHub Pages** (static, shareable).
- Small **TS FFT** lib (e.g. fft.js) for k-space → image.
- Optional: **Three.js** if vtk.js glyph animation feels clunky for many precessing arrows.
- Optional: React for the control panel (or plain TS).
- Self-contained: own `package.json`, `node_modules/`, `dist/` (all gitignored) inside `mri-sim/`.

## Architecture (engine ⟂ renderer — MVP)
- **Model (pure TS, no rendering):** the simulation engine.
  - `SpinSystem` — grid of magnetization vectors (Mx,My,Mz) + proton density (from the phantom).
  - `Sequence` — timeline of events: RF pulse, slice/phase/freq gradients, readout windows.
  - `Simulator` — steps the state (precession at local field = B₀ + gradients; RF tips; optional T1/T2).
  - **Honest k-space:** a readout sample = **Σ over spins of density·e^{iφ}**, where φ is the
    phase each spin accrued from the gradients = **the DFT computed as a sum over spins**.
    Fill k-space this way → inverse FFT → image. (This is why #3 "physics" and #4 "honest"
    are one mechanism — the spins carrying phase *is* the transform.)
  - Pure + testable (unit-test: known phantom → spins → k-space → FFT → recovers phantom).
- **View (vtk.js + 2D panels):** draws current model state — spin glyphs/arrows, bore, B₀;
  2D k-space panel; 2D image panel; sequence-diagram timeline. No physics in the view.
- **Presenter/Controller:** wires UI controls → sequence/sim params; runs the animation loop;
  pushes model state → view each frame.

Renderer choice: **vtk.js** (relevance + skills); swap-able behind the view interface if
arrow-animation perf demands Three.js later.

## Scene elements
- **Magnet / bore** — static geometry.
- **B₀** — field direction arrow (along z).
- **Proton spins** — grid of arrow glyphs; precess at the Larmor rate; tip on RF.
- **RF pulse** — visualized tip of the net magnetization.
- **Gradients** — slice-select (tilt frequency along an axis; highlight the excited slab),
  phase-encode (row-wise phase twist), frequency-encode (readout).
- **k-space panel** — 2D grid, fills line-by-line as phase-encode steps advance.
- **Image panel** — 2D, updates via live inverse FFT as k-space fills.
- **Sequence-diagram timeline** — RF + 3 gradient channels over time, scrubbable.

## Interactions
- Play / pause / **scrub** the pulse sequence.
- Sliders: TR, TE, flip angle, gradient strength, matrix size.
- Toggle: show one slice vs the slab; show/hide k-space & image panels.
- Step one TR at a time (watch one k-space line get written).

## Milestones (MVP-first)
- **M0** — scene scaffold: bore, B₀ arrow, grid of spin arrows along z.
- **M1** — animate precession + RF tip (90°) + relaxation.
- **M2** — slice-select: gradient → position-dependent frequency; RF excites only the
  matching slab (highlight); only slab tips.
- **M3** — phase-encode (row twist) + frequency-encode read; show the signal waveform.
- **M4** — k-space fills line-by-line over repeated TRs; **live inverse FFT → image**.
- **M5** — controls + sequence-diagram timeline.
- **Later** — oblique slices (combine gradients), bSSFP steady-state mode + banding demo,
  3D volume render, multi-slice.

## Accuracy / honesty
- **Educational simplification first** — kinematic spins + the Fourier relationship, not a
  full Bloch solver. **Label it as such** in the UI/README (no overclaiming physical exactness).
- Optional later: a **Bloch-accurate mode** (numerically integrate the Bloch equations) for
  realism — a clear "advanced" toggle.

## Out of scope
- Clinical/patient data (this is abstract/conceptual — **not** a DICOM viewer).
- Real reconstruction pipelines, real scanner hardware fidelity.

## Relation to systole (the applied pipeline)
Same portfolio lane (medical imaging), different concern: this is the **physics/education**
piece; the systole pipeline proper is the **applied ML** piece (seg→EF). Kept as a
self-contained subfolder in the systole repo.

## Don't reinvent blind
Look at existing sims for inspiration/validation (Hargreaves' Bloch demos, the classic
"Bloch Simulator", IMAIOS animations) — but build your own; that's where the learning is.
