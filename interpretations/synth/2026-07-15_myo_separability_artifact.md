# The "myo needs texture" gap was a diagnostic artifact (deform-misalignment)

**Date:** 2026-07-15 · **Task:** synth generation fidelity · **Beads:** f4hk (premise invalidated), 04bh (finding retracted)

## TL;DR

The long-standing finding that *synth myocardium is under-separated from blood and needs within-slice
texture* (f4hk, from 04bh: "myo|cav per-slice d′ 0.21× real") is a **measurement artifact**, not a
generator defect. The fidelity diagnostic painted synth with `deform=0.15` (a training augmentation that
warps the anatomy) but measured per-class intensities against the **un-warped** mask. The thin myo ring
is maximally sensitive: the warped ring samples adjacent bright blood + dark background, manufacturing a
huge spurious per-class spread. With `deform=0` (anatomy aligned to the measurement mask), the gap
inverts:

| metric (per-slice, myo↔blood) | deform 0.15 (buggy) | deform 0 (correct) | real |
|---|---|---|---|
| LV-myo\|LV-cav d′ | 0.63 (0.24×) | **4.54 (1.71×)** | 2.65 |
| LV-myo\|RV d′ | 0.83 (0.31×) | **4.53 (1.71×)** | 2.66 |
| LV-myo distance shape-W1 | 0.729 (worst class) | **0.072** | — |

Measured correctly, synth myo is **over**-separated from blood (d′ 1.7× real) and its within-class
distribution **already matches** real (shape-W1 0.07). Synth is too *clean/easy*, not under-textured.

## What actually caused it

`SynthPainter.synthesize_from_labels(mask, cfg, …)` warps `mask` internally when `cfg.deform>0` and
returns the warped mask. The fidelity tool (`SynthFidelity.separability/distance/variance`, `Render`)
discarded the returned mask and indexed synth pixels by the **original** `Y`. With a ±0.15 elastic warp,
the painted heart slips off `Y`; for a thin ring (myo) most "myo" pixels then fall on blood/bg. Bigger
blobs (RV/cav, bg) are far less perimeter-sensitive, so the artifact masqueraded as a *myo-specific*
texture defect. `deform` is a **training diversity knob** (invent new anatomy) — it has no place in an
appearance comparison, whose entire premise is "paint the *same* anatomy, compare the paint."

## Corrected fidelity picture (deform off)

Within-class **shape** (mean-centered distribution) now matches real for every class — shape-W1: myo
0.07, cav 0.10, RV 0.13. **All remaining gaps are LOCATION (mean brightness / contrast):**

| class | distance location (mean gap) | render mean±σ (real → synth) |
|---|---|---|
| RV | 0.51 | +1.64±0.55 → +1.88±0.90 |
| LV-myo | 0.42 | +0.24±0.27 → **−0.52**±0.51 (too dark) |
| LV-cav | 0.41 | +2.23±0.72 → +1.90±0.66 |

**Synth myo is painted too dark** (−0.52 vs real +0.24). That is exactly why synth *over*-separates myo
from bright blood (d′ 1.7×), and the one pair it now *under*-separates is myo|bg (0.90×) — myo pushed too
dark toward background. The real lever for generation fidelity is tissue **mean brightness/contrast**
(bSSFP signal levels / bg composition after per-image z-score), **not within-slice texture**.

## Fix

- `SynthFidelity._paint(mask, cfg, n_classes, real_img)` — single helper that forces `deform=0`
  (`model_copy`) before painting; all fidelity/separability/variance/distance calls route through it.
- `Render.render_synth_vs_real` paints with `deform=0.0` so the grid aligns to the mask.
- Also fixed an unrelated crash the diagnostic hit first: `synthesize_from_labels` typed `mask: Int`
  (signed) but `load_to_gpu` returns uint8 masks → widened to `Integer`.

## Consequences / open

- **04bh is retracted**, **f4hk's premise is invalid.** The "structured myo texture" direction is not
  supported by the data — synth within-class shape already matches real.
- Any prior conclusion drawn from this tool at `deform` default is suspect and should be re-checked
  (coverage/W1-based reads in the composite/fidelity lane).
- New (correctly-measured) question: **synth myo is ~0.76 z too dark** and synth is over-clean/over-
  separated. That's a contrast/brightness axis (tissue signal levels, bg composition) — related to
  machine-conditioned generation (ex1), not texture. Whether closing it moves real Dice is unknown and
  needs a retrain A/B (data-space fidelity ≠ Dice — cf. the coverage-neutral result).
