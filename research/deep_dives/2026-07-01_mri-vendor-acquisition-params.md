# MRI Vendor Acquisition Parameters for Cardiac Cine bSSFP — Machine-Conditioned Synth (`systole`)

**Date:** 2026-07-01
**Purpose:** Ground the cine-bSSFP signal painter in real, cited acquisition parameters per scanner vendor/model, replacing hand-guessed constants. Feeds `reference/acquisition.yaml` (unverified paper leaves) and the inflow-enhancement model.
**Scope:** Cardiac **cine balanced-SSFP** only (TrueFISP / balanced-FFE / FIESTA / True-SSFP). NOT LGE, NOT flow, NOT T1/T2 mapping.

---

## 0. Executive summary (headline numbers)

| Vendor | Sequence name | TR (ms) | TE (ms) | Flip 1.5T | Flip 3T | Confidence |
|--------|---------------|---------|---------|-----------|---------|------------|
| Siemens | TrueFISP | 2.7–3.5 | ~1.2–1.5 | 60–80° | 30–50° | High (family), Med (per-model) |
| Philips | balanced-FFE (bFFE) | 2.7–3.4 | ~1.2–1.5 | 60° (40–80) | ~45° | Med |
| GE | FIESTA | 3.0–4.0 | 1.2–1.7 | 45–60° | 30–50° | Med |
| Canon/Toshiba | True-SSFP | ~3.0–3.5 | ~1.2–1.5 | ~50–60° | ~40° | Low (sparse literature) |

**One-line physics headline:** TR/TE are near-vendor-invariant (hardware-gradient-limited, all cluster ~3 ms / ~1.3 ms). **The real vendor+field axis of variation is FLIP ANGLE**, driven by the SAR ceiling: **~80° max at 1.5T, ~50° max at 3T** for TR≈3 ms (Wang et al., escholarship 0tm0772f / PubMed 26509846). Recommend the YAML vary flip per (vendor, field), and treat TR/TE as a shared prior.

**Inflow headline:** flowing blood reads BRIGHTER than the bSSFP steady-state equation predicts because fresh unsaturated spins enter the slice each TR. Proposed sweep for a "fresh-blood fraction" boost: **f_fresh ∈ [0.0, 0.6]**, from the geometric model `f = min(1, v·TR/thk)` with cardiac through-plane v ≈ 5–90 cm/s, thk = 6–8 mm, TR ≈ 3 ms.

---

## 1. Our datasets → scanner models → field strength

The vendors/models in the task are **exactly** the M&Ms-2 scanner roster plus ACDC. Confirmed mapping:

### ACDC (Bernard et al. 2018, IEEE-TMI; CREATIS challenge page)
> "1.5 T (Siemens **Aera**) and 3.0 T (Siemens **Trio Tim**)." Sequence: "SSFP … short axis," slice 5 mm (sometimes 8 mm), 5 mm gap, in-plane 1.37–1.68 mm²/px, 28–40 frames.
Source: <https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html>. **No TR/TE/flip published** — must borrow from Siemens family prior.

### M&Ms-1 (Campello et al. 2021, IEEE-TMI)
375 subjects, 4 vendors: Siemens (dom A, n=95), Philips (dom B, n=125), GE (dom C, n=50), Canon (dom D, n=50). Sites in Spain + Germany. Voxel 0.85×0.85×10 → 1.45×1.45×9.9 mm. Mostly 1.5T. Paper documents vendor + resolution but **does not tabulate per-vendor TR/TE/flip**.
Source: <https://www.researchgate.net/publication/352494774> ; <https://www.ub.edu/mnms/>.

### M&Ms-2 (Martín-Isla et al. 2023, IEEE-TMI) — **the model-level roster**
> "nine scanners from three vendors … Siemens: **Avanto (AVA), Avanto Fit (AVF), Symphony (SYM), SymphonyTim (SYT), TrioTim (TRT)**; Philips: **Achieva (ACH)**; GE: **Signa Excite (EXC), Signa Explorer (EXP), Signa HDxt (HDXT)**." **"Most images … 1.5T and a small fraction of 3.0T."**
Source: <https://link.springer.com/chapter/10.1007/978-3-030-93722-5_36> ; <https://www.researchgate.net/publication/370071927>.

### Field-strength table for the named models (well-established from vendor specs)

| Model | Vendor | Field | Note |
|-------|--------|-------|------|
| Symphony / SymphonyTim | Siemens | **1.5T** | legacy platform |
| Avanto / Avanto Fit | Siemens | **1.5T** | very common cine workhorse |
| Aera | Siemens | **1.5T** | ACDC 1.5T arm |
| Trio / TrioTim | Siemens | **3.0T** | ACDC 3T arm |
| Vida | Siemens | **3.0T** | newer 3T |
| Skyra | Siemens | **3.0T** | (referenced in task) |
| Achieva | Philips | **1.5T and 3.0T** | sold in both; M&Ms Achieva mostly 1.5T |
| Signa HDxt | GE | **1.5T** (also a 3.0T variant existed) | M&Ms GE mostly 1.5T |
| Signa Explorer / Excite | GE | **1.5T** | |
| Canon/Toshiba (Vantage etc.) | Canon | 1.5T and 3.0T | model not named in M&Ms |

**Implication:** the overwhelming majority of our training data is **1.5T**. The 3T contingent is ACDC-Trio + the "small fraction" in M&Ms-2. So the 1.5T flip values are the high-prior ones; 3T is the tail.

---

## 2. Per-vendor cine bSSFP TR / TE / flip — evidence

### Vendor-invariant physics first (why TR/TE barely move)
Cine bSSFP wants the **shortest possible TR** to fit temporal-resolution and banding constraints; TE = TR/2 (symmetric). On modern gradients this floors at ~2.7–4 ms regardless of vendor. mriquestions: "TE … as short as possible … 1–2 ms range. RF flip angle made large (80–90°)." SCMR 2020 gives only **temporal resolution ≤45 ms** and **slice 6–8 mm** — no vendor TR/TE/flip, because "these vary among vendor platforms" (PMC7038611).

So: **TR ≈ 3 ms, TE ≈ 1.3 ms is a defensible cross-vendor prior.** The discriminating axis is flip.

### Flip angle: the SAR-limited axis (the load-bearing finding)
> "SAR limits the maximum allowable flip angle … to approximately **50° or less at 3T and 80° or less at 1.5T** (for a conventional bSSFP cine … TR of 3 ms)."
Wang / Nayak et al., *Free-breathing variable flip angle bSSFP cardiac cine with reduced SAR at 3T*: <https://pubmed.ncbi.nlm.nih.gov/26509846/>, <https://escholarship.org/uc/item/0tm0772f>. **Confidence: HIGH.**

Optimal-contrast studies:
- Max myocardial signal at 3T occurs at **flip 30–40°** (Prediction-of-myocardial-signal, PMC6570530; study on 3T GE Signa Excite HD, TR 5.0–5.9 ms).
- 1.5T optimal-contrast flips commonly **50–80°**; contrast-enhanced work sweeps 50/80/90/100° (Kuetting 2018, JMRI, PubMed 28429574).

### Siemens (TrueFISP)
- Family prior: TR 2.7–3.5, TE ~1.2–1.5. mriquestions worked example (1.5T, Siemens-style): TE 1.15, flip **90°**, thk 7 mm, reported TR ~38.5 ms *temporal* (segmented). Flip 90° is the aggressive-contrast end; routine clinical 1.5T Siemens cine is often **60–70°**.
- 3T (Trio/Vida): SAR-capped **~35–50°**.
- Confidence: HIGH for the family band, MEDIUM per-model (no per-model published cine tables found for Aera/Trio/Avanto specifically).

### Philips (balanced-FFE / bFFE, Achieva)
- Standard bFFE cine cited as **TR 2.7 / TE 1.2 / flip 40°** (low end) up to routine **60°**; 3D/whole-heart Achieva variants use higher flips (100°) but that's not 2D cine. Sources: mri-q / Kuetting; PLOS One whole-heart Achieva (TR 4.6/TE 1.8/FA 100 — coronary, excluded).
- Achieva sold at both 1.5T and 3T; our M&Ms Achieva data mostly 1.5T.
- Confidence: MEDIUM.

### GE (FIESTA, Signa)
- 1.5T Signa cine FIESTA: **TR 3.0–4.0, TE 1.2–1.7, flip 45–60°** (typical clinical). 3D-FIESTA example TR 3.5–4.0 / TE 1.6–1.7 / flip 50°.
- 3T (Signa Excite HD, from PMC6570530): TR 5.0–5.9 ms, flip swept 10–90° with **optimum 30–40°**.
- Confidence: MEDIUM (1.5T), MEDIUM (3T, one strong paper).

### Canon / Toshiba (True-SSFP)
- Sparse literature. No dedicated cine-parameter paper surfaced. By hardware parity with the field, assume **TR ~3.0–3.5, TE ~1.3, flip ~50–60° @1.5T, ~40° @3T**.
- Confidence: **LOW** — flag as extrapolated from cross-vendor physics, not a Canon-specific citation. Canon is only 50/375 subjects (M&Ms-1 dom D) and 1.5T-dominant, so error here is low-weight.

---

## 3. Proposed `reference/acquisition.yaml` block

**Granularity decision:** Evidence supports **per-field flip** (1.5T vs 3T is real and cited) but NOT robust per-model tables. TR/TE are shared priors. So: encode **one TR/TE per vendor** + **flip split into `flip_deg_1p5t` / `flip_deg_3t`**. Per-model granularity is NOT warranted by current evidence — leave a comment for future refinement (Aera/Trio are the only model-resolved pair we could even attempt, via ACDC's field split).

All leaves `verified: false` (inert until human check). `value` uses the mid of the cited band; put the band in `based_on`.

```yaml
acquisition:
  # NOTE: TR/TE are cross-vendor physics priors (~3ms/~1.3ms, gradient-limited);
  # the discriminating axis is flip_deg, SAR-capped ~80@1.5T / ~50@3T (PubMed 26509846).
  # Per-model granularity NOT warranted by evidence — vendor+field only.
  Siemens:                 # TrueFISP; Aera/Avanto/Symphony=1.5T, Trio/Vida=3T
    tr_ms:        {value: 3.0,  source: "mriquestions cine-parameters; SCMR 2020 (PMC7038611)", based_on: "cross-vendor cine bSSFP prior, TR 2.7-3.5ms", extracted_by: paper, verified: false}
    te_ms:        {value: 1.3,  source: "mriquestions cine-parameters", based_on: "TE~=TR/2, 1.15-1.5ms", extracted_by: paper, verified: false}
    flip_deg_1p5t:{value: 70,   source: "Kuetting 2018 JMRI (PubMed 28429574); mriquestions", based_on: "1.5T routine cine 60-80deg, 90deg aggressive-contrast", extracted_by: paper, verified: false}
    flip_deg_3t:  {value: 40,   source: "Wang/Nayak (PubMed 26509846); PMC6570530", based_on: "3T SAR cap <=50deg; opt contrast 30-40deg", extracted_by: paper, verified: false}
  Philips:                 # balanced-FFE (bFFE); Achieva 1.5T & 3T
    tr_ms:        {value: 3.0,  source: "mri-q; Kuetting 2018", based_on: "bFFE cine TR 2.7-3.4ms", extracted_by: paper, verified: false}
    te_ms:        {value: 1.3,  source: "mri-q", based_on: "1.2-1.5ms", extracted_by: paper, verified: false}
    flip_deg_1p5t:{value: 60,   source: "mri-q; Kuetting 2018 (PubMed 28429574)", based_on: "bFFE cine 40-80deg, routine ~60", extracted_by: paper, verified: false}
    flip_deg_3t:  {value: 45,   source: "Wang/Nayak (PubMed 26509846)", based_on: "3T SAR cap <=50deg", extracted_by: paper, verified: false}
  GE:                      # FIESTA; Signa HDxt/Explorer/Excite=1.5T
    tr_ms:        {value: 3.5,  source: "3D-FIESTA cine reports; PMC6570530 (3T Signa)", based_on: "FIESTA cine TR 3.0-4.0ms (1.5T), 5.0-5.9ms (3T)", extracted_by: paper, verified: false}
    te_ms:        {value: 1.5,  source: "FIESTA cine reports", based_on: "1.2-1.7ms", extracted_by: paper, verified: false}
    flip_deg_1p5t:{value: 55,   source: "3D-FIESTA cine reports; mriquestions", based_on: "1.5T FIESTA cine 45-60deg", extracted_by: paper, verified: false}
    flip_deg_3t:  {value: 40,   source: "PMC6570530 (3T GE Signa Excite HD)", based_on: "3T opt contrast 30-40deg, SAR cap <=50", extracted_by: paper, verified: false}
  Canon:                   # True-SSFP; Toshiba/Canon. LOW confidence — extrapolated.
    tr_ms:        {value: 3.2,  source: "cross-vendor physics extrapolation", based_on: "no Canon-specific cine paper found; hardware parity prior", extracted_by: paper, verified: false}
    te_ms:        {value: 1.4,  source: "cross-vendor physics extrapolation", based_on: "TE~=TR/2", extracted_by: paper, verified: false}
    flip_deg_1p5t:{value: 55,   source: "cross-vendor physics extrapolation", based_on: "assume Siemens/GE-like 50-60deg; LOW confidence", extracted_by: paper, verified: false}
    flip_deg_3t:  {value: 40,   source: "cross-vendor physics extrapolation (SAR cap)", based_on: "LOW confidence", extracted_by: paper, verified: false}
```

> **Note on the code's read shape:** task says code does `ref.get("acquisition", <vendor>, "tr_ms"/"flip_deg")`. If the reader expects a flat `flip_deg` (not `_1p5t`/`_3t`), collapse to a single `flip_deg` per vendor using the 1.5T value (since data is ~1.5T-dominant) and drop the field split — but that discards the one well-cited axis of variation. **Recommend updating the reader to take a field-strength arg** and index the split. Flagging for the integrator.

### Confidence flags per value
- **Well-established:** SAR flip caps (80/50), TR~3ms/TE~1.3ms cross-vendor floor, ACDC/M&Ms scanner rosters + field strengths.
- **Estimated (paper band, mid picked):** all specific `flip_deg` values, GE/Philips TR/TE.
- **Extrapolated (no direct cite):** entire Canon block.

---

## 4. Inflow / entry-slice enhancement — physics + sweep

### Why blood is brighter than the bSSFP steady-state equation predicts
The steady-state signal `M_ss` assumes spins have seen many RF pulses (reached SSFP). **Blood flowing through the 2D slice violates this:** each TR, some in-slice blood is flushed out and replaced by **fresh, fully-relaxed, unsaturated spins** that have NOT been driven down to the low steady state. Fresh spins give a much larger transverse signal on excitation → **bright blood** ("flow-related enhancement" / "entry-slice phenomenon"). Confirmed by mriquestions (why-gre-flow-signal, time-of-flight-effects) and xrayphysics mr_flow.

Conditions that maximize it (all present in cardiac cine): short TR, moderate-large flip, thin 2D slice, **flow perpendicular to slice** (through-plane). Effect is much easier to reach than TOF flow-voids because it scales with **TR (long) not TE (short)**.

### Quantitative geometric model (the sweep lever)
Standard first-order model (Bernstein *Handbook of MRI Pulse Sequences*; textbook TOF):

```
f_fresh = min(1,  v · TR / thk)
```
Fraction of the slice replaced by unsaturated blood per TR. When `v·TR ≥ thk`, the slice is fully refreshed each TR (`f=1`, maximal enhancement, blood essentially at its fully-relaxed signal). Then the painted blood signal is a convex blend:
```
S_blood_painted = (1 - f_fresh)·S_ss  +  f_fresh·S_fresh
```
with `S_fresh` = the single-excitation (unsaturated) signal ≫ `S_ss`. A pragmatic generator simplification is a **multiplicative boost** `S_blood *= (1 + β·f_fresh)` and sweep β/f jointly.

### Plugging cardiac numbers (to bound the sweep)
- **Slice thickness** thk = **6–8 mm** (SCMR; ACDC 5–8 mm) → use 7 mm.
- **TR** ≈ 3 ms.
- **Through-plane blood velocity v** in short-axis cine is NOT the valve peak; it's the ventricular through-plane component. Handles:
  - Mitral inflow E-wave ~**90 cm/s**, A-wave ~**80 cm/s** (4D-flow vs Doppler, PMC9843884) — upper bound near valve planes / basal slices.
  - Aortic-root longitudinal motion ~**10 cm/s** (Nature s41598-021-83278-x) — low end (annular tissue, not jet).
  - Mid-ventricular cavity through-plane bulk flow is far slower than valvular jets — order **5–30 cm/s** in most short-axis slices, spiking at base during rapid filling/ejection.
- Per-TR displacement at v=90 cm/s, TR=3 ms: 0.9 cm/s × 0.003 s = **2.7 mm** → f = 2.7/7 ≈ **0.39**. At v=30 cm/s → f ≈ 0.13. At v=5 cm/s → f ≈ 0.02.

Note segmented cine acquires k-space over MANY heartbeats/TRs; the effective per-*phase* refresh integrates over the segment, so real enhancement can exceed the single-TR f — hence propose the sweep top out **above** the single-TR max.

### Proposed physiological sweep
- **f_fresh ∈ [0.0, 0.6]** (0 = distal/slow slices, ~0.4 single-TR at fast basal flow, headroom to 0.6 for segmented accumulation + banding/partial-volume brightening).
- Equivalent parameterization: draw v ~ U(5, 90) cm/s per slice, thk from the acquisition prior, TR from vendor prior, compute f = min(1, v·TR/thk), then boost. **Make it slice-position-aware** (basal > apical) if the generator has that context.
- **Confidence:** model = well-established (textbook); the exact cardiac through-plane v distribution = **estimated** (valve-plane peaks well-cited, mid-cavity through-plane values are inferred). Sweep, don't pin.

---

## 5. Open gaps / follow-ups
- No per-model published cine TR/TE/flip for Aera/Trio/Avanto/Achieva/HDxt individually — values are family+field priors. A human could pull DICOM headers from the actual ACDC/M&Ms volumes to *verify* (that's what `verified:false` is waiting for).
- Canon block is pure extrapolation — lowest trust, lowest data weight.
- ACDC/M&Ms papers document vendor+resolution+field but NOT sequence timing — DICOM-header mining is the only route to dataset-true numbers.

---
### Sources
- ACDC challenge: <https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html>
- M&Ms-1: <https://www.researchgate.net/publication/352494774> ; <https://www.ub.edu/mnms/>
- M&Ms-2 roster: <https://link.springer.com/chapter/10.1007/978-3-030-93722-5_36>
- SAR flip caps (3T bSSFP): <https://pubmed.ncbi.nlm.nih.gov/26509846/> ; <https://escholarship.org/uc/item/0tm0772f>
- 3T myocardial signal / optimal flip: <https://pmc.ncbi.nlm.nih.gov/articles/PMC6570530/>
- Flip optimization 1.5T (gadobutrol): <https://pubmed.ncbi.nlm.nih.gov/28429574/>
- SCMR 2020 protocols: <https://pmc.ncbi.nlm.nih.gov/articles/PMC7038611/>
- Cine parameters (mriquestions): <https://mriquestions.com/cine-parameters.html>
- Inflow / GRE flow signal: <https://mriquestions.com/why-gre-uarr-flow-signal.html> ; <https://mriquestions.com/time-of-flight-effects.html> ; <http://xrayphysics.com/mr_flow.html>
- Mitral inflow velocities: <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9843884/>
