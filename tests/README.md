# Tests — the pyramid

```
        e2e / visual smoke   cardioview/web + mri-sim scripts/*-smoke   ← thin, MANUAL, asserts pixels
      integration            tests/integration/*                        ← module pairs / pipeline chains
   unit                      tests/unit/*                               ← equivalence-class, fast (base)
```

Wide base, thin top: many cheap unit tests, fewer integration tests, a handful of
manual visual smokes. A bug should be caught by the cheapest layer that can see it.

## unit/ — equivalence-class

One representative per **equivalence class** of inputs (plus boundaries), not exhaustive
cases. Partition the input space by behaviour — e.g. for `fit_square`: *larger than target*
(crops), *smaller* (pads), *equal* (identity); for EF: *EDV > ESV* (normal), *EDV == 0*
(NaN guard). Test one example from each class and the edges between them. Fast, deterministic,
no data, no GPU.

## integration/ — module-pair / pipeline chains

If modules A and B sit in the same pipeline, the **A → B chain** must work on the same
inputs/outputs the units promise: A's output is a valid input to B, and the chain produces
what each unit guarantees. These catch interface drift that per-unit tests miss (shape, dtype,
label, spacing conventions across a boundary).

- `test_pipeline_chain.py` — data-free synthetic chains: `preprocess → dataset`
  (resample → square-fit), `measure ↔ evaluate` (one mask pair feeds both EF and Dice/surface).
- `test_smoke.py` — full chain on **real ACDC** (load → identify LV → measure → Dice).
  Skips when the (gated, out-of-repo) dataset is absent, so a fresh clone still passes.

## e2e / visual smoke (manual)

Per-viewer, not here: `cardioview/web` and `mri-sim` each ship a headless screenshot smoke
that asserts pixels (catches render bugs — invisible glyphs, blank scenes — a unit test can't).
Manual, not in CI. See each project's own `tests/` notes.

---
**Layout.** This `tests/` tree covers the **cardioseg** pipeline (Python). The viewers keep
their own suites next to their code — `cardioview/tests/`, `cardioview/web/tests/`,
`mri-sim/tests/` — same pyramid, run with each project's test command. Run cardioseg's:
`python -m pytest tests/ -q`.
