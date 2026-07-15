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

## Reproduce
- Calibration: `python -m cardioseg.evaluation ef_calibrate` → `plots/ef_calibration.json`.
- Source-loss A/B: `python -m cardioseg.training.train --split static_main --ef-lambda 0.02 [--ef-kaggle] --quick`.
