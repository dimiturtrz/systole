# CardioSAM Triage — GO/NO-GO on Running It On Our Data

**Date:** 2026-06-30
**Scope:** Code-availability + provenance gate (NOT a methods review). Verdict on whether we can run "CardioSAM" on our own cardiac MRI, plus citation verification.
**Method:** Web search + WebFetch against primary source (arxiv HTML/abstract).

## Verdict (one line)

**NO-GO. CODE GATE = NONE.** Real preprint exists and the ~0.934 ACDC number checks out as stated, but it is a **single-author arxiv preprint with no code, no weights, no venue, no peer review, and no cross-vendor evaluation** — and it carries credibility red flags. Not runnable; not a trustworthy baseline.

---

## 1. PRIMARY SOURCE

"CardioSAM" **is the real published title** (not a paraphrase).

- **Title:** *CardioSAM: Topology-Aware Decoder Design for High-Precision Cardiac MRI Segmentation*
- **Author:** Ujjwal Jain (**single author**)
- **Venue:** **arXiv preprint only** — cs.CV. No conference/journal. Not peer-reviewed.
- **ID / date:** arXiv:2604.03313, submitted **2026-03-31**.
- **Links:** https://arxiv.org/abs/2604.03313 · https://arxiv.org/html/2604.03313v1 · https://arxiv.org/pdf/2604.03313

The arxiv ID resolves (abstract + HTML render both load), so the preprint genuinely exists — it is not web hearsay or a hallucinated reference. The matching descriptors ("topology-aware", frozen-SAM-encoder + cardiac-specific trainable decoder, ACDC ~93.4 Dice) confirm this is **the** model referenced in the earlier second-hand research.

**Architecture (as claimed):** frozen SAM encoder + lightweight trainable cardiac-specific decoder, with (a) a Cardiac-Specific Attention module said to inject "anatomical topological priors" and (b) a Boundary Refinement Module. This matches the "topology-aware SAM-for-cardiac" framing.

**Other real SAM-for-cardiac candidates considered** (none claim topology-aware + 0.934 ACDC, so none displaces CardioSAM as the match):
- *Temporal-Spatial Adaptation of Promptable SAM for cine CMR* (arXiv:2403.10009; MICCAI/Springer 2024) — promptable SAM, not topology decoder.
- *SAMba-UNet: SAM2 + Mamba UNet for Cardiac MRI* (arXiv:2505.16304).
- *U-MedSAM* (uncertainty-aware MedSAM, evaluated on ACDC).
- General MedSAM / SAMed / SAM-Med2D adaptations (see SAM4MIS survey, github.com/YichiZhang98/SAM4MIS).

## 2. NUMBER VERIFICATION

**~0.934 is confirmed as the *as-stated* number** in the primary source (subject to the trust caveats below).

- Headline: **mean Dice 93.39%** across cardiac structures on ACDC (≈ the "0.934" from earlier research). Also reported: IoU 87.61%, pixel accuracy 99.20%, HD95 4.2 mm.
- Per-structure (Table VII, per WebFetch of the HTML): **LV cavity 94.2%, RV cavity 92.1%, myocardium 93.8%.**
- Claimed **+3.89% Dice over nnU-Net** (next best), p<0.001 paired t-test.
- **In-distribution:** trained and tested on ACDC.

**Caveat — confidence is LOW that this number means anything.** It is self-reported in a non-peer-reviewed single-author preprint with no code to reproduce it. The 93.39% is plausible-looking (ACDC in-distribution SOTA sits in the low-to-mid 90s) but unverifiable independently. Treat as "the paper claims 0.934", not "0.934 is established."

## 3. CODE AVAILABILITY — THE GATE → **NONE**

- (a) Pretrained weights downloadable? **No.**
- (b) License? **None** (no repo).
- (c) Framework / SAM checkpoint? **Not specified** in any accessible artifact.
- (d) Inference entrypoint / README showing prediction on a new NIfTI? **None — no repository exists.**
- (e) Last-commit recency / issues? **N/A — no repo.**

No GitHub or HuggingFace repository found via targeted search (`CardioSAM github`, `"Ujjwal Jain" CardioSAM code release`). The paper itself contains **no Code-availability, no Data-availability statement, no GitHub URL, and no weights link** (verified by WebFetch of both abstract and full HTML). Multiple unrelated "Ujjwal Jain" GitHub profiles exist; none hosts CardioSAM.

**Classification: NONE (paper only).** Reproducing would require a full from-scratch reimplementation of an unreviewed single-author method — explicitly out of scope for a "can we run it" gate.

## 4. CROSS-VENDOR / OOD

**Not reported.** The paper evaluates **only on ACDC (single-vendor, in-distribution)**. No M&Ms, no M&M-2, no cross-dataset, no unseen-vendor results anywhere in the abstract or full HTML. This is the decisive weakness for our purposes: the well-known failure mode is in-distribution SAM-cardiac models collapsing to ~0.75 on unseen vendors (cf. our own landscape note: ~42-47% Dice drop Siemens→Philips). CardioSAM offers **zero evidence** it survives vendor shift. Its headline 93.4% is therefore not comparable to a cross-vendor target.

## Credibility Red Flags (provenance)

- **Single author**, no institutional co-authors, no venue, no peer review.
- **Buzzword without substance:** abstract claims integrating **Particle Swarm Optimization** "to navigate the hyperparameter manifold," but PSO implementation details are absent from the methodology (per HTML fetch) — unsupported by technical content.
- Suspiciously high secondary metrics (pixel accuracy 99.20%) and a tidy "+3.89% over nnU-Net" headline with no released code to check.
- No code, no weights, no OOD test — the exact pattern of a thin, unverifiable preprint.

## Bottom Line

Real preprint, real title, number confirmed *as self-reported*, but **NO-GO**: not runnable (CODE = NONE), not peer-reviewed, single-author, no cross-vendor evidence. Do not adopt as a baseline or cite as a SOTA result without heavy "unverified single-author preprint" caveats. If we want a runnable SAM-cardiac baseline, pursue MedSAM / SAMed / SAM4MIS-listed repos instead — those have public code.

### Sources
- https://arxiv.org/abs/2604.03313 (abstract — title, author, date, no code statement)
- https://arxiv.org/html/2604.03313v1 (full HTML — Table VII per-structure Dice, PSO claim, no OOD, no code/data availability)
- https://github.com/YichiZhang98/SAM4MIS (SAM-for-medical survey — alternative runnable candidates)
- https://arxiv.org/abs/2403.10009 ; https://arxiv.org/abs/2505.16304 (other real SAM-cardiac candidates considered)
