# mri-sim roadmap

Tracked in beads (`bd show <id>`). This file is the human-readable view.

## Done — pipeline complete, end-to-end
- 3D proton grid precessing on cones (individual-spin view), batched into one polydata
- Slice-selective RF flip (completes within the slice-select window); tiltable/oblique slab
- Larmor (MHz) selects slice height; realistic TR/TE in ms
- Phase encode actually winds in-slice spins into a position-dependent ramp; steepness steps each TR (ky loop)
- Frequency encode / readout; gradient coloring by local Larmor
- k-space fills one line per TR → live inverse-FFT image
- Pulse-sequence diagram (zoomed to the encode window; relaxation tail fast-forwarded)
- One shared clock; log-scaled speed; precession runs at a steady real-time rate
- Scene legend; centralized color palette (legend can't drift)
- Pure-TS engine, 42 tests + asserting visual smoke

## Planned
Roughly ordered by value. The first three form one arc: make the signal *real*.

1. **T2\* dephasing + FID/echo signal decay** — `cardiac-seg-clq`
   Spins fan out from off-resonance after excitation → net signal decays (FID),
   recovering at the echo. Motivates *why TE matters*.
2. **Receive signal trace panel (FID/echo waveform)** — `cardiac-seg-xit`
   Plot the measured signal over a TR: FID after RF, decaying, gradient echo at TE.
   Closes the loop spins → measured signal → k-space. (Depends on #1.)
3. **Full 2D k-space acquisition (ky stepping)** — `cardiac-seg-por`
   Step ky across TRs, fill the 2D grid, reconstruct via 2D iFFT; map the
   phase-encode wind-up steepness to the ky line being filled.
4. **Real ACDC slice as phantom (vtk reslice)** — `cardiac-seg-w2n`
   Replace the synthetic disk with a real short-axis ACDC slice. Where vtk.js
   earns its keep (vtkImageData + reslice); ties the sim to the cardiac-imaging data.
5. **Tissue contrast (T1/T2 weighting via TR/TE)** — `cardiac-seg-wpq`
   Per-region T1/T2 so TR/TE change image *contrast* (T1w / T2w / PD) — the clinical knob.
