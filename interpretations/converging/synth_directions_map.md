# Synth → real: directions map (living tracker)

**Purpose:** single source for "where are we" on the synth-generalization effort. Updated as directions move.
Each row = maturity tag + best number + verdict + next lever. Keep it honest; keep it current.

**North star (owner):** domain generalization via **all-synthetic training**. **Zero-real is always the big goal.**
Everything else supports it or is a diagnostic for it.

---

## THE GOAL — A. Zero-real GENERATE (synth mask + synth color)

**Best: ~0.61 test** (composition-fixed) · real-trained 0.854 · **blocker (CORRECTED 2026-07-16 by looking at
predictions): RV OMISSION in a patient-clustered tail, NOT a uniform color collapse.**

Pooled RV 0.457 was a **bimodal artifact**: median per-case RV (mid-slice) = **0.853** (real-trained level),
but a tail of ~7–10% of cases (patient-clustered: P004/P011/P014) score RV **0.0** — the model segments the LV
**perfectly** (myo ring + cav) and **entirely omits the RV** (0 RV pixels predicted) though the RV blood pool is
large and clearly visible. Not mis-colored, not confused with LV, not mis-bounded — **absent.** This is a
**detection / RV-shape-coverage / recall** failure, NOT color (the LV color is fine). Confirms why the contrast
levers were dead — the problem was never contrast. Viz: `scratchpad/rv_fail.png` (mechanism), rebuild via
`scratchpad/rv_fail_viz.py`. Sub-lever tree:

| lever | status | evidence |
|---|---|---|
| composition / whole-FOV levels | ✅ **WIN +0.059** (on) | fixed myo z −0.52→correct (2gk5) |
| color fidelity (spread / blood-mean / spectrum) | ❌ dead | nk70, xmcf — fidelity≠Dice, ×5 |
| contrast diversity (tissue_spread width, contrast_random) | ❌ **refuted** | nttu — widen flat, randomize *hurts* RV; RV needs contrast *structure* not diversity |
| shape coverage (composite / pathology pool) | ❌ dice-neutral | coverage 0.78→0.94, Dice flat |
| noise / high-freq aug | 🔒 load-bearing (constraint) | nk70.1 — can't lower (→0.385) |
| **RV detection / recall** (why RV fires 0 px on a patient tail) | ⬜ **UNTRIED — now the prime suspect** | omission mode; is RV under-weighted / under-represented in synth batches? |
| **RV-shape coverage** (the failing configs) | ⬜ **UNTRIED** | do synth RV shapes cover the omitted patients' RV? |
| within-label heterogeneity (multi-Gaussian paint) | ⬜ deprioritized | it's a *color* lever; the blocker isn't color |
| **learned shape prior** (pathology tail) | ⬜ filed (vpn5) | reaches the 22% DCM/HCM tail SSM misses |

**Frontier redirected by the failure viz:** the blocker is RV **omission** on a patient tail, so the levers are
**RV detection/recall** and **RV-shape coverage**, NOT color/heterogeneity. Open disambiguation: is the tail a
recall problem (model under-fires RV), a coverage problem (these RV shapes absent from synth), or an
appearance-quirk of specific scans? Characterize the failing patients (all-slices? common vendor/motion/RV-size?)
to pick the lever.

---

## Diagnostic for A

**B. Zero-real REPAINT** (real mask + synth color) — **~0.68**, measured. Not deployable; it's the *color-axis
isolator*: color costs 0.17, shape 0.07. Told us the A blocker is mostly color, not shape.

---

## Support (secondary to A)

| # | direction | best | maturity | note |
|---|---|---|---|---|
| C | Real + synth **AUGMENTATION** (mix) | ~0.84 | under-measured | does synth help *on top of* real? (pwih) — no clean matched A/B yet |
| D | Controlled / **INVERSE / twin** | — | not started | fit color to a scan; project's 2nd named direction (ncph) |
| E | **EF lane** | cal −4pp | mature ✅ | calibration (tb58), Kaggle weak-sup (2b7m); own objective |
| F | nnU-Net baseline | — | reference | quarantined SOTA ref |

---

## Where we are right now (2026-07-16)

A is the goal and it's stuck at ~0.61. The **color/contrast sub-levers are exhausted** (nk70, xmcf, nttu). The
frontier is the **three untried A-levers** — within-label heterogeneity, RV-appearance, learned shape prior —
plus the deeper question of an RV floor. **Next:** take the untried levers seriously, warm-started for speed;
start with within-label heterogeneity (distinct, cheap). Do **not** pivot off A.
