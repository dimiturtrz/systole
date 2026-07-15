# EF defensibility — three levers on the biased number (2026-07-15)

## The problem
The segmentation masks are good (mean Dice ~0.86–0.88) but the **derived** EF is biased: the flagship
under-predicts EF by a systematic ~−6pp on the ACDC val centre and ~−10pp on unseen vendors, with wide
limits of agreement. The mechanism is a **systematic ES-cavity under-fill** — the model slightly
under-segments the small end-systolic blood pool, which inflates ESV's relative weight and drags
`EF = (EDV−ESV)/EDV` down. Per-pixel Dice is nearly blind to this (a thin boundary shell is a small
Dice hit but a large ESV-ratio hit).

Three levers were measured against it — one post-hoc, two at the source.

## 1. Post-hoc linear calibration — `tb58` (the cheapest lever)
Fit `ef_corr = a·ef_pred + b` on VAL (held-out ACDC centre), apply once to TEST. No retrain.
`python -m cardioseg.evaluation ef_calibrate`.

Flagship fit `ef_corr = 1.103·ef_pred + 2.10` (n=150):

| axis | MAE | bias | note |
|------|-----|------|------|
| val (ACDC) | 7.1 → **5.4** | −6.4 → **0.0** | in-sample (fit here) |
| Canon (unseen) | 9.9 → **5.4** | −9.8 → **−2.9** | transfers |
| GE (unseen) | 11.0 → **7.4** | −10.1 → **−3.2** | transfers |

The key result is **transfer**: the EF bias is vendor-systematic, so a single linear fit carries to
unseen vendors — unlike temperature scaling (`calibrate.py`), which is domain-shift-limited and does
*not* transfer. Residual −3pp on test = the correction under-shoots the larger OOD shift (post-hoc
calibration stays domain-shift-limited; it removes the shared systematic component, not the vendor-
specific remainder). Free, no retrain, no Dice cost by construction.

## 2. Volume-consistency loss — `ax4a` core (source-level)
A differentiable soft EDV/ESV from the LV-cav probability mass → dimensionless Huber vs GT, folded as
a nudge into the seg gradient (`VolLoss.vol_loss`, `VolConsistency`; Kendall `ef_learn` or fixed
`ef_lambda`). Converged 3-seed A/B on `static_main`: Dice 0.846 → 0.852 (free), **EF MAE 11.2 → 9.2**
(~2pp, marginal 1.8σ). Small but free — it supervises the volume that Dice ignores.

## 3. Kaggle EF-only weak supervision — `2b7m` (source-level, no masks)
96 pooled Kaggle DSB2015 cines with **EF labels but no masks** → EF-RATIO Huber (spacing-invariant),
segment-summed over each cine's full SAX stack (`KaggleEF`). Matched A/B, seed 0, `static_main`,
`--quick` (only difference = `--ef-kaggle`):

| arm | val EF MAE | test EF MAE | val Dice | test Dice |
|-----|:---:|:---:|:---:|:---:|
| vol-consistency only | 7.8 | 11.0 | 0.883 | 0.841 |
| **+ Kaggle EF-only** | **5.9** | **9.6** | 0.887 | 0.847 |

EF MAE −1.9pp val / −1.4pp test, Dice free (slightly up). Same direction on two independent test sets +
Dice-neutral + sound mechanism → a real signal. **Promising, not hardened**: single-seed + `--quick`,
so the absolute magnitude is soft, but the delta is fair (identical 40-epoch budget both arms controls
undertraining; abs test 11.0 matches the known `static_main` baseline). Multi-seed confirmation was
deferred this session for efficiency.

## Verdict
- **Calibration is the cheapest defensible lever** — free, no retrain, removes the bias outright on the
  fit centre, and *transfers* to unseen vendors (−1.7 to −4.5pp MAE). It is the floor everything else is
  measured against.
- **The two source losses attack the bias mechanistically** (they move the masks, not just the reported
  number) and each buys ~1.5–2pp EF MAE for free on Dice. They are **stackable with calibration** — one
  fixes the shared systematic component post-hoc, the other reduces the ES-cavity error that generates
  it — so the honest next step is calibration *on top of* a source-loss model, not either alone.
- Caveats, stated: the calibration val bias→0 is in-sample; the Kaggle result is single-seed; the three
  arms sit on different splits/budgets (each is a matched *internal* A/B, not one clean ladder — don't
  read the absolute numbers across sections as a single sequence).

## 4. The number with a CI — `qhdm` (defensibility)
A point EF MAE is not defensible; a reviewer can't tell a solid number from a lucky draw of nine
patients. So every reported EF MAE/bias carries a **percentile bootstrap 95% CI** (resample the held-out
EF *pairs* with replacement, `Measure.bootstrap_ef_ci`, seed-pinned) — the honest error bar on a single
split, no retrain. k-fold retrain would add *training* variance too, but the binding uncertainty here is
the **thin test split**, which the bootstrap exposes directly and cheaply.

Calibrated flagship, 95% CI on MAE and bias:

| axis | n | MAE (cal) | MAE 95% CI | bias (cal) | bias 95% CI |
|------|:--:|:--:|:--:|:--:|:--:|
| val (ACDC) | 150 | 5.4 | [4.5, 6.4] | −0.0 | [−1.3, 1.2] |
| Canon | 9 | 5.4 | **[2.2, 9.7]** | −2.9 | [−8.0, 1.7] |
| GE | 69 | 7.4 | [6.1, 8.9] | −3.2 | **[−5.3, −1.1]** |

What the CIs actually say — three defensible statements the point estimates could not make:
- **Calibration removes the in-sample bias for real:** val bias CI [−1.3, 1.2] straddles 0. On the fit
  centre the correction is genuine, not a rounding artefact.
- **The residual OOD bias is real, not noise — on the powered split:** GE bias CI [−5.3, −1.1] **excludes
  0**. Post-hoc calibration is domain-shift-limited (it under-corrects the larger unseen-vendor shift),
  and at n=69 that residual is statistically resolved, not a guess.
- **Canon (n=9) is underpowered — say so, don't over-claim:** MAE CI [2.2, 9.7] is enormous and the bias
  CI straddles 0. At nine patients we cannot resolve our own EF error to better than a factor of ~4.

### The honest gap vs nnU-Net, with CIs
nnU-Net (same split): Canon EF MAE 2.6, GE 4.3. Ours calibrated: Canon 5.4, GE 7.4.
- **GE (n=69, powered):** nnU-Net's 4.3 sits **below** our CI [6.1, 8.9] → the gap is real and
  significant. We are genuinely behind on EF here — the model-class epistemic gap (a 1.6 M-param 2D U-Net
  vs a 92 M nnU-Net), traded for a 57× smaller deployable ONNX model. Stated, not hidden.
- **Canon (n=9, underpowered):** nnU-Net's 2.6 falls **inside** our CI [2.2, 9.7] → at this n we *cannot*
  claim we are worse than nnU-Net on Canon. The honest read is "indistinguishable at n=9", not a number.

This is the Gate-2 defensibility bar: a number with a CI, the bias/LoA reported not buried, and a gap vs
SOTA that is quantified and owned. The CI is what converts "EF MAE ~11%" from a boast-or-apology into a
claim with stated power.

## Reproduce
- Calibration + CI: `python -m cardioseg.evaluation ef_calibrate` → `plots/ef_calibration.json`
  (per-axis MAE/bias, uncalibrated vs calibrated, each with a bootstrap 95% CI).
- Source-loss A/B: `python -m cardioseg.training.train --split static_main --ef-lambda 0.02 [--ef-kaggle] --quick`.
