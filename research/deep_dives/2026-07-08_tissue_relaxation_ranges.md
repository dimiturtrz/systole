# Tissue MRI Relaxation Parameter Ranges (T1/T2/PD) at 1.5T & 3T — for Domain-Randomized Synthesis

**Date**: 2026-07-08
**Status**: settled (myocardium/blood/fat/liver/lung well-covered; skeletal muscle T2-spread thinner)
**Supersedes**: none

## TL;DR

Native myocardial T1 spread is real inter-subject/inter-study biology + method variance: 1.5T ~950–1050 ms, 3T ~1150–1300 ms; T2 ~45–56 ms (1.5T) / ~45–52 ms (3T). Blood T2 is the strongly field-dependent axis (~250→~150→~68 ms as 1.5T→3T→7T, and drops further venous/deoxygenated). Sample per-image with a truncated-normal at the in-vivo mapping mean, clipped to the min/max range below; treat the range AS the spread where studies disagree.

## Question

Compile primary-source T1/T2/PD central values AND their biological/measurement spread (mean±SD or min–max range) for myocardium, blood, fat, lung, liver, skeletal muscle, at both 1.5T and 3.0T, so tissue params can be sampled per synthetic image over the physically-plausible manifold instead of ad-hoc jitter.

## Reference table

PD is relative (water=1.0), approximate, from tissue water-content literature; ranges are coarse. All T1/T2 in ms. **"Range" = spread across studies/subjects** (the quantity you want), not a CI on one mean.

### 1.5 T

| Tissue | T1 mean (range) | T2 mean (range) | PD (~) | Primary cite |
|---|---|---|---|---|
| **Myocardium (native)** | ~1000 (950–1050) | ~50 (45–56) | 0.80 | in-vivo mapping [S3][S6][S7]; T1 995.8±30.9, T2 55.8±2.8 [S3] |
| Blood (arterial/cavity) | ~1440 (1350–1550) | ~250 (200–290) | 0.87 | [S4] ASL; in-vitro [S1]; oxygenation-dep [S5] |
| Fat (subcutaneous/epicardial) | ~290 (250–380) | ~130 (100–165) | 1.0 (lipid) | [S2][S8] |
| Lung parenchyma | ~1100 (800–1300) | T2* ~2.1±0.3 (T2 ~40–50) | 0.1–0.2 | [S9] (T2* UTE); [S2] |
| Liver | ~570 (500–650) | ~55 (40–102) | 0.70 | [S1][S2] |
| Skeletal muscle | ~1010 (870–1100) | ~45 (35–50) | 0.75 | [S1][S2] |

### 3.0 T

| Tissue | T1 mean (range) | T2 mean (range) | PD (~) | Primary cite |
|---|---|---|---|---|
| **Myocardium (native)** | ~1180 (1150–1300) | ~50 (45–52) | 0.80 | T1 1183.8±37.5, T2 51.6±3.0 [S3]; 1160–1227 clusters [S6]; ~1193±39 [S7] |
| Blood (arterial/cavity) | ~1650 (1550–1800) | ~150 (100–165) | 0.87 | [S4][S10]; in-vitro [S1]; T2 field-drop [S5] |
| Fat (subcutaneous/epicardial) | ~380 (370–450) | ~130 (150–220 apparent) | 1.0 (lipid) | [S1][S2][S8] |
| Lung parenchyma | ~1200 (1000–1400) | T2* ~0.74±0.1 (≤0.5) | 0.1–0.2 | [S9] |
| Liver | ~810 (700–900) | ~42 (30–50) | 0.70 | [S1][S2] |
| Skeletal muscle | ~1410 (1300–1450) | ~50 (30–50) | 0.75 | [S1][S2] |

## Findings

- **Myocardium is the best-characterized tissue and the highest-confidence numbers.** In-vivo mapping in 58 healthy volunteers: T1 995.8±30.9 (1.5T) / 1183.8±37.5 (3T); T2 55.8±2.8 (1.5T) / 51.6±3.0 (3T) [S3]. Standardization/clustering work found normal T1 clusters at 958±16 & 1027±19 (1.5T) and 1160/1067/1227 (3T) [S6], and ~1193±39 at 3T [S7]. The requested "~950–1050 (1.5T) / ~1150–1300 (3T)" is confirmed as the inter-study envelope.
- **(b) Myocardial T1 SD is dominated by METHOD/sequence, not biology.** Explicitly flagged in the literature: normal MOLLI-T1 SDs vary ~11-fold across protocols; dependencies on TE/TR/FA differ between 1.5T and 3T, "thwarting meaningful T1 standardization even within a single field strength" [S6]. MOLLI vs ShMOLLI differ by ~100 ms on the same subjects (MOLLI 943±45 vs ShMOLLI 833±55 septum, 1.5T) [S7b]. Real biological modifiers exist but are smaller: sex (women higher, ~1211 vs 1173 at 3T [S7]), heart rate (higher HR → longer T1, shorter T2 [S3]); age/BMI negligible [S3]. **Implication for sampling: the ±150 ms 3T envelope is mostly a sequence-choice axis — good, because a domain-generalizing model SHOULD be invariant to it.**
- **(a) Blood T2 is the strong field-dependence flag.** Arterial/oxygenated blood T2 falls steeply with B0: ~250 ms (1.5T) → ~150 ms (3T) → ~68 ms (7T at Hct 0.42) [S5]. R2 rises faster with field and with deoxygenation; venous/deoxygenated blood T2 is far shorter (~20 ms at 7T; ~40–60% O2 vs 99–100% arterial) [S5]. Hematocrit is a minor axis vs oxygenation [S5]. Blood T1 rises with field (~1440→~1650 ms, linear in B0) [S4] like other tissues — it is T2 that carries the anomalous field dependence. **For LV-cavity (oxygenated) use the high end; if simulating venous return / RV, T2 should drop.**
- **Blood T1 central values**: in-vitro/ASL literature clusters ~1440 (1.5T) and ~1550–1650 (3T) for arterial [S1][S4]; in-cavity in-vivo measurements run higher (2076±125 at 3T reported for ventricular blood [S10]) — reflecting measurement context (partial volume, flow). Range spans ~1550–2100 at 3T depending on method; treat the wide spread as the sampling range.
- **Fat T1 is short and rises with field** (~290→~380 ms) [S1][S8]; T2 is temperature-sensitive (apparent T2 ~100→190 ms at 1.5T, ~130→220 at 3T with heating) [S8] — at body temp use ~100–130 ms. Fat PD (lipid protons) ≈ tissue water density; treat as ~1.0 on a lipid-signal basis.
- **Lung parenchyma is dominated by T2*, not T2.** T2* ~2.1±0.3 ms (1.5T) → ~0.74±0.1 ms (3T), can be ≤0.5 ms at 3T [S9], from air–tissue susceptibility, not intrinsic T2. Low PD (~0.1–0.2) from ~80% air. For a bSSFP/cine synth this tissue is near-signal-void; the wide T1 range is poorly constrained (secondary).
- **Liver & skeletal muscle** follow the canonical Stanisz field trend (T1 +41% liver, +40% muscle 1.5T→3T [S1]); Bojorquez review confirms wide inter-study liver T2 (40–102 ms) [S2].

## Open questions

- **Skeletal-muscle and lung T1 spreads are the weakest** — thin primary sourcing for the min/max; ranges are estimates. Fine for a low-weight/background tissue, but don't treat as tightly grounded.
- **Exact Stanisz 2005 table numbers not verified against the original.** The paper is paywalled (Wiley 402); the numbers attributed to [S1] here (3T: blood T1~1441/T2~290, heart T1~1471/T2~47, liver T1~812/T2~42, muscle T1~1412/T2~50) are the *widely-cited* Stanisz in-vitro values but were surfaced via search-echo, not read from the source — **medium confidence, and in-vitro at 37°C (ex-vivo animal tissue for heart/liver/muscle; human blood), so systematically offset from in-vivo.** For myocardium/blood prefer the in-vivo mapping numbers [S3][S4][S6][S7], which are higher-confidence for a cine-MRI synth.
- Epicardial vs subcutaneous fat relaxation difference not separately sourced (assumed similar).

## How to sample (domain randomization)

- **Per synthetic image, draw each tissue's T1/T2 independently** (or with light correlation) as **truncated normal**: mean = the in-vivo mapping mean for the target field; SD chosen so ±2σ ≈ the min/max range; **hard-clip to [min, max]**. This concentrates mass at physiological central values while still covering the tails — matches "physically-constrained diversity," not flat jitter.
- **Alternative: uniform over [min, max]** where you want maximal coverage of the plateau (justified for tissues whose "spread" is really inter-study/method disagreement, e.g. myocardial T1 — the model *should* be invariant across that whole band). Uniform is the more aggressive DR choice; truncated-normal the more calibrated one. Consider uniform for myo-T1 (method axis) + truncated-normal for T2 (tighter biology).
- **Couple field strength as a switch**: sample field ∈ {1.5T, 3T} first, then draw from that field's row. This bakes the correct T1-rises-with-field and blood-T2-drops-with-field correlations for free instead of decorrelated per-param noise.
- **Blood: link T2 to an oxygenation latent** — LV cavity high-O2 (long T2, use top of range), RV/venous low-O2 (short T2). Don't sample cavity T2 independent of chamber.
- Don't sample lung by T2 — set it near signal-void (low PD + very short T2*) or exclude from the relaxation model.

## Sources

- [S1] Stanisz et al., "T1, T2 relaxation and magnetization transfer in tissue at 3T", Magn Reson Med 2005;54(3):507-512 — PMID 16086319. (in-vitro, 37°C; canonical table. Exact values medium-confidence, not read from paywalled original.) accessed 2026-07-08
- [S2] Bojorquez et al., "What are normal relaxation times of tissues at 3 T?", Magn Reson Imaging 2017;35:69-80 — systematic review; gives inter-study ranges (e.g. liver T2 40–102, fat T2 41–371). https://sciencedirect.com/science/article/abs/pii/S0730725X16301266 — accessed 2026-07-08
- [S3] Reiter et al., "Comparison of native myocardial T1 and T2 mapping at 1.5T and 3T in healthy volunteers", Wien Klin Wochenschr 2018 — PMC6459801 (n=58): T1 995.8±30.9/1183.8±37.5, T2 55.8±2.8/51.6±3.0. accessed 2026-07-08
- [S4] Zhang et al., "In vivo blood T1 measurements at 1.5T, 3T, and 7T", Magn Reson Med 2013 — PMID 23172845 (blood T1 linear in B0; 7T ~2.1s). accessed 2026-07-08
- [S5] blood-T2 oxygenation/field literature (Zhao et al. "Oxygenation and hematocrit dependence of transverse relaxation rates of blood at 3T"; "Dependence of blood T2 on oxygenation at 7T", PMC3971007) — arterial T2 ~68 ms @7T Hct 0.42, steep field & O2 dependence. accessed 2026-07-08
- [S6] "Standardization of T1-mapping ... clustered structuring for benchmarking normal ranges", Int J Cardiol 2020 — clusters 958±16/1027±19 (1.5T), 1160/1067/1227 (3T); 11-fold SD variation, method-dominated. accessed 2026-07-08
- [S7] Age-gender native T1 reference (3T ~1193±39; women>men) — PMC4044551. accessed 2026-07-08
- [S7b] Piechnik et al. MOLLI vs ShMOLLI age-gender reference (1.5T & 3T) — PMC3560015: MOLLI 943±45 septum 1.5T; ShMOLLI 833±55. accessed 2026-07-08
- [S8] fat T1/T2 field & temperature dependence (subcut lipid T1 ~250→380–450 ms 1.5T→3T; apparent T2 temp-dependent) — musculoskeletal 3T relaxation lit + [S2]. accessed 2026-07-08
- [S9] "Comparison of Lung T2* During Free-Breathing at 1.5T and 3.0T with UTE", PMC3122137 — T2* 2.11±0.27 ms (1.5T) / 0.74±0.1 ms (3T). accessed 2026-07-08
- [S10] Myocardial & blood T1 quantification in normal volunteers at 3T — PMC3106825 / JCMR (ventricular blood T1 2076±125 @3T, in-cavity context). accessed 2026-07-08
