# Dataset acquisition + annotation conventions (ACDC, M&Ms-1, M&Ms-2, MSD)

**Date**: 2026-06-20
**Status**: grounded (web sources cited; some per-patient field-strength splits unconfirmed)
**Why**: fill the reference store (acquisition.yaml / conventions.yaml) that feeds normalization,
stratified eval, and the model card. Resolves NOTICE_BOARD open questions on papillary convention
+ ACDC acquisition.

## TL;DR — the papillary question is answered, and it deflates the bias theory (honestly)

**ACDC, M&Ms-1, and M&Ms-2 all use the SAME LV-cavity convention: papillary muscles +
trabeculae are INCLUDED in the LV blood pool** (not carved out as myocardium). M&Ms-2 *explicitly*
states it was annotated "following the ACDC annotation standards." M&Ms-1 is from the same
Barcelona-led consortium with the same three-structure protocol.

**Implication for our −5.6% cross-dataset EF bias:** it is *probably NOT* a papillary/label-protocol
mismatch — the convention is shared. So the bias is genuinely **domain/intensity generalization**
(contrast differs across vendors → segmentation shifts → volume shifts), which points back at
**harmonization (N4 / Nyúl)** and **calibration**, not protocol documentation. A mildly deflating
but important result: it kills the "easy" explanation and keeps the honest one.
(Caveat: convention *stated* the same; per-annotator drift at apex/base still exists — that's the
irreducible inter-observer floor, not a systematic offset.)

## Acquisition specs

| dataset | n | centres | vendors | field | scanners | structures | LV label |
|---|---|---|---|---|---|---|---|
| **ACDC** | 100 | 1 (Dijon, FR) | Siemens | 1.5T + 3T | Aera 1.5T, Trio Tim 3T | LV/RV/Myo | **3** |
| **M&Ms-1** | 375 | 6 (ES×4, DE, CA) | Siemens, Philips, GE, Canon | 1.5T + 3T* | multiple | LV/RV/Myo | **1** |
| **M&Ms-2** | 360 | 3 (ES) | Siemens, GE, Philips | 1.5T (343) + 3T (17)† | 9 scanners | LV/RV/Myo | **1** |
| **MSD Task02** | ~30 | — | — | — | — | left atrium only | — |

\* M&Ms-1 per-vendor/field split documented in the TMI paper tables (not fully pinned here).
† M&Ms-2 field split from our own `dataset_information.csv` (verified locally), not the paper.

- **ACDC**: single-centre University Hospital of Dijon; Siemens **Aera (1.5T)** + **Trio Tim (3T)**.
  5 pathology groups (NOR/DCM/HCM/MINF/ARV), 20 each. ED+ES annotated. Info.cfg carries
  height+weight (→ BSA available).
- **M&Ms-1** (Campello 2021, IEEE TMI 40(12):3543): 375 subjects, 6 centres across **Spain (4),
  Germany (1), Canada (1)**, **4 vendors** (Siemens/Philips/GE/**Canon**). Split: 150 labelled +
  25 unlabelled train, 200 test (incl. an unseen-vendor held-out slice). Each study annotated by
  the origin-centre clinician, then a 4-researcher pairwise SOP revision to cut inter-centre
  contour variance at apex/base. CSV has Age/Sex/Height/Weight → **BSA available**.
- **M&Ms-2** (2021): 360 subjects, **3 centres in Spain, 9 scanners, 3 vendors**
  (Siemens/GE/Philips — note: **no Canon**, unlike M&Ms-1). SA **+ LA 4-chamber** views. ED+ES.
  Sequential split 160/40/160. **Annotated following ACDC standards.**
- **MSD Task02_Heart**: left-atrium mono-structure MRI (Medical Segmentation Decathlon). Different
  task (LA, not LV/RV/myo) — not directly usable for the EF pipeline; parked.

## Papillary / trabeculae convention (the priority target)

- **ACDC**: "trabeculae and papillary muscles are **included in the ventricular cavity**" — the
  endocardial border is traced around them (they count as blood pool). Standard CMR-challenge
  convention. (Clinically, whether to include them in EF is a known methodological variable with
  real impact — biggest in LVH/HCM — but the *dataset ground truth* fixes "included.")
- **M&Ms-1 / M&Ms-2**: same three-structure protocol; M&Ms-2 explicitly "following ACDC annotation
  standards." → **shared convention.**
- **Net**: convention is consistent across our train (M&Ms-2) and test (ACDC) → not the bias source.

## What this means for the open work

- `4yf` (papillary check): largely **answered by literature** — convention is shared. The geometric
  probe is now a *confirmation* (cheap), not a discovery. Bias hypothesis → redirect to domain/intensity.
- `qfz` (N4/Nyúl harmonization) + `lnd` (calibration): **these remain the real EF levers** — the
  bias is domain-driven, exactly their target.
- `reference/` store: acquisition.yaml can be filled now (values above, with these citations);
  `verified: true` for cited specs, `verified: false` for the unconfirmed per-patient field splits.
- BSA indexing: available for **ACDC + M&Ms-1** (height+weight), not M&Ms-2 → ragged, as expected.

## Open / unconfirmed
- M&Ms-1 exact per-vendor and per-field-strength counts (in the TMI paper tables; not extracted here).
- ACDC per-patient 1.5T vs 3T split (paper says both scanners used; per-patient mapping not public).
- M&Ms-2 letter→vendor mapping in our CSV uses full names already (Siemens/Philips/GE) — fine.

## Sources
- [ACDC challenge](https://www.creatis.insa-lyon.fr/Challenge/acdc/) · Bernard et al. 2018, IEEE TMI (scanners: Siemens Aera 1.5T / Trio Tim 3T, Dijon)
- [M&Ms-1 paper, IEEE Xplore](https://ieeexplore.ieee.org/document/9458279/) · Campello et al. 2021, TMI 40(12):3543 (375 subj, 6 centres ES/DE/CA, 4 vendors)
- [openmedlab M&Ms resource](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/M&Ms.md) (375 patients, 6 centres, 4 vendors, labels LV=1/MYO=2/RV=3)
- [M&Ms-2 challenge](https://www.ub.edu/mnms-2/) (360 subj, 3 centres ES, 9 scanners, 3 vendors, follows ACDC standards)
- ACDC papillary convention: trabeculae + papillary included in cavity — multiple CMR segmentation refs (e.g. [Frontiers review](https://www.frontiersin.org/journals/cardiovascular-medicine/articles/10.3389/fcvm.2020.00025/full))
