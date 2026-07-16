# Synth → real: directions map (living tracker)

**Purpose:** single source for "where are we" on the synth-generalization effort. Updated as directions move.
Each row = maturity tag + best number + verdict + next lever. Keep it honest; keep it current.

**North star (owner):** domain generalization via **all-synthetic training**. **Zero-real is always the big goal.**
Everything else supports it or is a diagnostic for it.

---

## THE GOAL — A. Zero-real GENERATE (synth mask + synth color)

**Best: ~0.61 test** (composition-fixed) · real-trained 0.854 · **located blocker: RV/myo COLOR collapse**
(RV ~0.44–0.50, gross mislocalization; myo ~0.47–0.55; cav ~0.72 nearly fine).

This is the number the project lives or dies on. The gap is **color-dominant, thin-structure** (RV free wall,
myo ring); cav (bright blob) survives. Sub-lever tree:

| lever | status | evidence |
|---|---|---|
| composition / whole-FOV levels | ✅ **WIN +0.059** (on) | fixed myo z −0.52→correct (2gk5) |
| color fidelity (spread / blood-mean / spectrum) | ❌ dead | nk70, xmcf — fidelity≠Dice, ×5 |
| contrast diversity (tissue_spread width, contrast_random) | ❌ **refuted** | nttu — widen flat, randomize *hurts* RV; RV needs contrast *structure* not diversity |
| shape coverage (composite / pathology pool) | ❌ dice-neutral | coverage 0.78→0.94, Dice flat |
| noise / high-freq aug | 🔒 load-bearing (constraint) | nk70.1 — can't lower (→0.385) |
| **within-label heterogeneity** (multi-Gaussian paint) | ⬜ **UNTRIED** | distinct spatial mechanism; the other half of SynthSeg's recipe |
| **RV-specific shape + appearance** (thin wall, RV–lung boundary) | ⬜ **UNTRIED** | repaint RV caps ~0.60 → residual is appearance, not contrast |
| **learned shape prior** (pathology tail) | ⬜ filed (vpn5) | reaches the 22% DCM/HCM tail SSM misses |

**Frontier = the three UNTRIED levers.** Open question underneath: is RV crackable by the generator at all, or
does zero-real cardiac have an RV floor (real-trained also drops RV most OOD, 0.854→0.587)?

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
