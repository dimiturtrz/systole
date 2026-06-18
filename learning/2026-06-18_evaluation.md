# Phase D hands-on · evaluation / validation rigor (Track E)

Connecting `materials/common/E_evaluation-theory.md` to the pipeline
(`cardioseg/evaluation/`: evaluate.py metrics, distribution.py KDE+Bland-Altman).
Opened by the learner's own insight: *every metric is a summary statistic of an
error distribution; the distribution + agreement is the truth.*

Grounding artifacts (held-out val, runs/acdc): per-class boundary-distance KDE
(LV-cav ASSD 0.5 / HD95 2.1mm tight; RV ASSD 1.6 / HD95 10mm broad) and EF
Bland-Altman (bias −1.5%, 95% LoA [−8, +5]).

Lesson plan:
1. The framing — metrics are summaries; look at the distribution ← here
2. E1 overlap vs boundary (Dice vs HD95/ASSD) — recap from G4
3. E2 metric pitfalls (Metrics Reloaded): pick metrics by failure mode
4. E3 Bland-Altman: bias + limits of agreement; MAE ≠ LoA
5. E4 calibration / uncertainty
6. E5 domain shift / clinical-grade gap

---

## Lesson E (framing) + E3 — agreement, not just an average

**The framing.** Dice, HD95, EF-MAE are each **one number = a summary statistic** of an
underlying **error distribution**. Different distributions collapse to the same number;
the number can't tell systematic from sporadic. So the move is: **plot the distribution
(KDE) and the agreement (Bland-Altman)**, then read summaries off it — not instead of it.

### Boundary-distance KDE (already built, distribution.py)
Per class, pool every surface-point distance pred↔GT (symmetric) → its shape:
- **LV-cav:** tall spike at 0 (ASSD 0.5, HD95 2.1mm) — tight, accurate.
- **RV:** low + **broad** (ASSD 1.6, HD95 10mm) — inconsistent boundary, *visibly* the weak class.
Same story Dice told (RV worst), but you **see** the spread, not just a worse scalar.

### E3 — Bland-Altman (EF agreement)
Comparing predicted EF to GT EF is a **method-agreement** question, not an accuracy %.
Plot per patient: x = mean(GT, pred), y = (pred − GT). Read:
- **bias** = mean(pred − GT) — the **systematic** offset. Ours: **−1.5%** (model under-
  estimates EF — most points below zero).
- **limits of agreement** = bias ± 1.96·SD — where 95% of patients land. Ours:
  **[−8.0, +5.0]**.

**Why this matters — it caught what MAE hid.** We reported **EF MAE 3.1%** → looks under
the ±5% clinical bar → "fine." Bland-Altman reveals **LoA [−8, +5]** — *wider* than ±5,
**plus** a −1.5% systematic bias. Individual patients can be 8 EF points low — enough to
cross a treatment threshold (HFrEF <40 vs HFmrEF 41–49). **MAE averaged the spread away;
agreement shows it.** (Clinical equivalence wants LoA inside ~±5; we're not there.)

**MAE vs LoA (the repo's flagged error).** MAE = mean |error| (one scalar). LoA = the
*spread* of signed error (bias ± 1.96·SD). The README compared MAE to a LoA-style ±5
threshold — apples to oranges. The fix is to report the Bland-Altman bias + LoA.

**Takeaway.** Evaluation isn't a leaderboard number — it's the **distribution** (KDE: is
RV broad?) and the **agreement** (Bland-Altman: is EF biased? how wide are the limits?).
Our model: cavity tight, RV broad, EF systematically −1.5% low with LoA too wide for
clinical equivalence. That's a far more honest verdict than "Dice 0.87, MAE 3.1%."

## Lesson E4 — uncertainty / calibration

**The gap:** the model gives a prediction but no "**how sure am I?**" For clinical use
you must know *when to trust it* (flag shaky cases for a human).

- **Confidence vs calibration.** Softmax gives a probability, but nets are usually
  **mis-calibrated** — say 0.99 when right only ~80% of the time (overconfident). A
  *calibrated* model that says 0.8 is right 80% of the time. Measured by reliability
  diagrams / **ECE** (expected calibration error).
- **Getting uncertainty:**
  - **softmax entropy** — high entropy / low max-prob at a voxel = uncertain (and it
    concentrates **at the boundary** — the hard part again).
  - **MC dropout** — keep dropout ON at inference, run N times → variance = uncertainty.
  - **ensembles** — N models; their disagreement = uncertainty (nnU-Net's 5-fold).
  - **test-time augmentation** — predict on augmented copies; variance = uncertainty.
- **Use:** a per-voxel **uncertainty map** (boundary glows) + a per-**case** score → flag
  low-confidence cases for human QC. That flag is the clinical safety valve — and it's
  how you **detect out-of-distribution** input (bridge to E5).
- **Us:** we output bare argmax, **no uncertainty**. A real tool needs the "I'm unsure
  here" channel.

## Lesson E5 — domain shift / the clinical-grade gap

**The core gap.** ACDC = **one hospital, ~one scanner, curated**. A model can ace ACDC
val and **fail** on:
- different **vendor** (Siemens/Philips/GE — different noise, reconstruction),
- different field strength / protocol / slice thickness,
- different **population** (obese, paediatric, congenital, post-surgical).
Empirically Dice drops **5–15%** absolute out-of-distribution. The **M&Ms** challenge
(multi-centre, multi-vendor) is the benchmark for this; ACDC alone **over-states** real
performance.

**Why a benchmark win ≠ a clinical tool:**
1. **Regulatory** — CE / FDA need *prospective* validation, not retrospective Dice.
2. **Reference quality** — single-expert GT (the E2 error-floor point).
3. **Edge cases** under-represented (pacemakers, congenital, poor image quality).
4. **Threshold decisions** — EF MAE 5% can flip HFrEF(<40)/HFmrEF for an individual.
5. **Audit trail** — explainability, failure-mode docs, continuous monitoring.

**Mitigations:** augmentation simulating scanner variability, domain adaptation /
instance norm, test-time adaptation, and **uncertainty/OOD detection** (E4) to flag
when you're off-distribution rather than silently guessing.

**The "hard 80%."** The demo (ACDC seg→EF) is the easy 20%; **robustness + validation +
regulatory is the hard 80%** — and the honest line this whole repo is built on.

**Takeaway (E4+E5).** A clinical-grade model knows **when it doesn't know** (calibrated
uncertainty → flag for review) and is **validated across scanners/sites** (not one
curated benchmark). We have neither — single split, single centre, no uncertainty. Our
honest claim is "competent on ACDC," explicitly *not* "clinical-grade."

### Quiz log
- [E1–E2 · 2026-06-18](quizzes/common/E1-E2_2026-06-18.md) — ~60% (rigor set). Concepts
  strong; mechanics (Dice vs IoU, thin-vs-fat) was the gap, drilled to solid.
- [E4–E5 · 2026-06-18](quizzes/common/E4-E5_2026-06-18.md) — ~80%. Uncertainty/calibration
  solid; sharpen: MRI is unit-less (intensity *distribution* shifts), clinical gap has
  multiple axes.
