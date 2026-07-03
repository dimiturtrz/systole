# Parametric / physics-based MRI generation for segmentation — state of the art

*2026-07-03. Web survey. Question: do people generate synthetic MRI the way we do (paint
anatomical labels with the bSSFP signal equation from tissue T1/T2/PD, sweep acquisition for
domain randomization), and does it work? Positioning our approach against prior art.*

**Our approach (recap):** paint label maps (SSM meshes / real masks) with the balanced-SSFP
steady-state signal equation using **literature** per-tissue T1/T2/PD (Stanisz 2005 / Bojorquez
2017); sweep TR / flip / field (1.5T/3T) for domain randomization; whole-FOV bg by tissue tiers;
cardiac; zero real images needed. Inverse = fit acquisition to a scan (digital twin). Measured:
zero-real synth ≈ 0.56 cross-vendor Dice vs ≈ 0.85 real-multi-vendor; synth best as augmentation +
unseen-domain.

---

## TL;DR — the field, and where we sit

- **The exact recipe exists and works, but mostly for BRAIN, and the strongest version uses
  per-subject qMRI maps, not literature values.** The closest published system is **UltimateSynth**
  (physics-based SynthSeg) — it beats random-contrast SynthSeg decisively. Validates our core bet
  (physics ≫ random GMM) but is brain + needs MR-Fingerprinting maps.
- **Cardiac + bSSFP-equation + literature-params + SSM shapes = our specific niche is largely
  unoccupied.** Cardiac physics synthesis papers are either inverse (Reverse Imaging) or GAN-based.
- **Our inverse/digital-twin direction was independently published Aug 2025** (Reverse Imaging) with
  strong numbers — validates the idea, and it hit the *same identifiability problem we did* (unknown
  flip angle). Partial scoop, but different framing (they use a diffusion prior; we fit parametrically).
- **The "diversity > fidelity" story is more subtle than our memory states.** The winning principle
  across the literature is **diversity constrained to the physically-plausible manifold** — random
  (SynthSeg) loses to physics (UltimateSynth), yet a single fidelity point loses to a physical sweep
  (our finding + general DR literature). Our physics-sweep sits exactly at the sweet spot.

---

## Q1 — SynthSeg lineage: how it makes intensities

**SynthSeg** (Billot, Greve, Van Leemput, Fischl, Dalca, Iglesias — *Medical Image Analysis* 2023;
arXiv 2107.09559): trained **only on label maps**, no real images. Intensities are sampled from a
**Gaussian Mixture Model conditioned on the segmentation** (classical Bayesian-segmentation
generative model) — each label gets a random Gaussian intensity per mini-batch, plus random
deformation, bias field, noise, resolution. This is **domain randomization, NOT physics**: the
sampled contrasts are unconstrained by MR signal equations, so many are physically impossible. That
is deliberate — force contrast-invariant features. Brain; **demonstrated to extend to cardiac MRI and
CT**. Ships in FreeSurfer.

→ Key point for us: SynthSeg = *random* label-conditioned contrast. We = *physical* label-conditioned
contrast. Same skeleton (paint labels, randomize), different intensity model. The next paper shows
the physical version wins.

## Q1b — **UltimateSynth: the physics-based SynthSeg** (the most important find)

*"UltimateSynth: MRI Physics for Pan-Contrast AI"* (PMC11661081, 2024). **Replaces SynthSeg's random
GMM with actual Bloch-equation physics.**
- Magnetization via spin-dynamics: system matrix A = R(T1,T2,TE)·Q(α,φ); builds a **122M-entry
  dictionary** over (T1,T2,PD)×(TR,TE,TI,flip), SVD-reduced to a 10-D contrast subspace (>99.95%
  energy) for efficient sampling of *physically valid* contrasts.
- **Tissue params from per-subject MR-Fingerprinting qMRI maps** (40 volunteers, 3T Prisma) — NOT
  literature values, NOT sampled distributions. Voxel-wise real T1/T2/PD.
- Brain only (16 labels). Trains nnUNetv2 purely on physics-synthetic, no test-time adaptation.
- **Beats SynthSeg**: Dice 0.83±0.07 vs 0.76 (synthetic test); on real ON-Harmony (3 vendors, 6
  scanners) 0.864 vs 0.850; **worst-case Dice 0.59 vs SynthSeg's 0.10**; lifespan volume MAE 3.5% vs
  8.6%. Authors' thesis: SynthSeg's "atypical contrasts unconstrained by MR physics" oversimplify
  boundaries; physics gives "natural tissue transitions."

→ **This is our thesis, proven (for brain): physics-constrained synthesis > random GMM.** Difference:
they need per-subject qMRI maps (expensive MRF acquisition); we use literature params + SSM shapes
(zero real qMRI). Our niche = the cheaper, cardiac, shape-generative version.

## Q2 — Bloch/physics simulators as training-data engines

- **MRXCAT** — cardiac-dedicated extended-cardiac-torso (XCAT) MR phantom. Heavily used to *evaluate*
  reconstruction/robustness (digital phantom, controllable), **rarely as a segmentation training
  source**. It's a single anatomical phantom → low anatomical diversity (the opposite of our SSM pool).
- **KomaMRI.jl** (Castillo-Passi et al., *MRM* 2023; arXiv 2301.02702) and **JEMRIS** — general
  Bloch solvers, GPU. Papers explicitly say "expected to be used for creating synthetic ML training
  data," but published *trained-a-segmentation-net* results are thin — mostly aspirational / used for
  reconstruction & sequence design, not seg training pipelines yet.

→ Full Bloch simulation is heavier than we need. Our closed-form bSSFP steady-state is the right
abstraction for cine (no transient/Bloch integration). No cardiac Bloch-sim-trained seg SOTA to beat.

## Q3 — bSSFP-equation label-painting for cardiac seg (our exact method)

Closest **forward** cardiac physics work is indirect. The cleanest match is in **brain/stroke**:
**Chalcroft, Crinion, Price, Ashburner — "Domain-Agnostic Stroke Lesion Segmentation Using
Physics-Constrained Synthetic Data"** (arXiv 2412.03318, 2024): synthesize multi-sequence MR by Bloch
simulation from **tissue parameter maps (T1/T2/PD)**, tissue params **sampled probabilistically from
literature-based distributions** (cites Tabelow 2019), train seg purely on synthetic, test real.
→ This validates BOTH (a) forward physics-label-painting for training and (b) **sampling tissue params
from literature mean±SD** — exactly our open task (bd `04bh`, jitter→physical param sampling). It's
brain/stroke, not cardiac cine. Nobody found doing bSSFP-equation cardiac cine painting from
literature params + SSM shapes = **our specific cell is open**.

## Q4 — qMRI / relaxometry-based synthesis

- **FASTR-SCANN** (Radiology: AI 2022, ryai.210294): cardiac **T1-mapping** seg via
  relaxation-based synthetic contrast — generates synthetic T1-weighted images along the T1 relaxation
  curve at varied inversion times → U-Net. Cardiac, relaxometry-driven, but for *T1 mapping*, not cine
  seg. Same spirit (signal model → synthetic contrasts for robustness), narrower target.
- SyntheticMR/MAGiC-style relaxometry modelling (PubMed 27753154) + multitask seg+relaxometry
  (arXiv 1911.12389) exist for brain. General principle (contrast from T1/T2 maps) is established;
  cardiac-cine + shape-generation is not their focus.

## Q5 — domain randomization vs adaptation, cross-vendor cardiac (M&Ms)

- **M&Ms challenge** (Campello et al., *IEEE TMI* 2021): 4 vendors (Siemens/Philips/GE/Canon), 6
  centres. **Winning approach = nnU-Net + heavy augmentation + batch norm** (Full et al.) — plain
  aggressive augmentation beat fancy adaptation. Histogram matching + z-score (Ma) also strong.
- Style-transfer DG (Random Style Transfer, arXiv 2008.12205; domain-adversarial, arXiv 2008.11776)
  and random-convolution DG (arXiv 2512.01510) are the DG-method family.
→ Matches our headline: **multi-vendor real + strong aug generalizes; the diversity is what matters.**
No published *pure-synthetic* method beats real-multi-vendor cross-vendor cardiac — consistent with
our 0.56(synth) vs 0.85(real-multi-vendor). Synthetic's role is augmentation / annotation-free /
unseen-domain, not replacement. Our honest framing is field-consistent.

## Q6 — digital twin / analysis-by-synthesis (our inverse direction)

**Reverse Imaging** (arXiv 2508.21254, Aug 2025) — the closest to our inverse/twin, and a partial scoop:
- Uses the **identical bSSFP steady-state equation** we do: f_SS = PD·sin(ω)/[1+cos(ω)+(1−cos(ω))·T1/T2].
- **Inverse**: infers spin properties (PD,T1,T2) from an observed bSSFP scan by posterior inference
  with a **diffusion prior** over spin properties (trained on mSASHA), then re-synthesizes arbitrary
  novel sequences for augmentation ("RI-Aug").
- Trained on ACDC (100, bSSFP cine); big gains generalizing to unseen sequences: on MOLLI, nnUNet
  baseline MYO 47.4% → **86.5%**, RV 24.0% → **87.0%**; on a device set MYO 86.9% → 93.1%.
- **Hit our exact identifiability wall**: "the exact FA for ACDC scans is unknown, we approximate it
  with ω=45°"; "does not provide precise spin property estimation." (We found single-frame bSSFP
  acquisition unidentifiable — same physics.)
→ Validates our inverse thesis and shows it works. Differentiators still open for us: **parametric**
fit (interpretable acquisition params, digital twin) vs their diffusion prior; and cardiac-cine cross-
*vendor* (they do cross-*sequence*). Worth citing as the state of the art we build against.

## Q7 — fidelity vs diversity (our "diversity > fidelity" memory, re-examined)

The literature splits the question we conflated:
- **General DR (robotics/sim2real):** very wide randomization *reduces peak performance* and can
  "remove task-critical structure"; **medium-fidelity + DR = best cost/performance** (Lil'Log DR
  survey; Benchmarking DR arXiv 2011.07112). One study: DR models had *worse* zero-shot than
  real-only. So unbounded randomization is not free.
- **UltimateSynth (medical):** physics-constrained (higher fidelity to the physical manifold) **beats**
  unconstrained random GMM — decisively.
- **Our finding:** the single most-accurate fidelity *point* trains *worse* than a physical *sweep*.

**Reconciled principle: diversity CONSTRAINED TO THE PHYSICALLY-PLAUSIBLE MANIFOLD wins.** Not random
(loses to physics), not a single fidelity point (loses to a sweep). Our physics-based acquisition
sweep is exactly this sweet spot — a stronger, more defensible framing than a flat "diversity >
fidelity." Update our internal note accordingly.

## Q8 — thin-structure / partial-volume fidelity

No paper found that specifically addresses **partial-volume realism of thin structures** in synthetic
MRI training data. UltimateSynth notes physics gives "natural tissue transitions" (implicitly better
boundaries) but doesn't quantify thin-wall PV. → Our myocardium over-spread finding (synth myo σ 4×
real, thin-wall PV) appears to be an **under-explored problem** — a potential contribution, not a
solved one.

## Q9 — sampling tissue T1/T2/PD from literature distributions

**Yes, known and validated** — Chalcroft (2412.03318) samples tissue params from literature-based
distributions for physics-constrained synthesis. UltimateSynth instead uses per-subject MRF maps
(the "gold" version). Post-hoc intensity jitter (what we currently do) is the *weaker* stand-in; the
literature endorses **sampling relaxation params from distributions** (our bd `04bh` direction). Good
prior-art support for that task.

---

## Methods table

| Method | Intensities from | Modality | Trained a net? | Headline result | Cite |
|---|---|---|---|---|---|
| **SynthSeg** | random GMM per label (no physics) | brain (→cardiac/CT) | yes, synth-only | contrast-agnostic; robust across contrasts/res | Billot 2023, MedIA; arXiv 2107.09559 |
| **UltimateSynth** | **Bloch physics** from per-subject **MRF qMRI maps** | brain | yes, synth-only | **Dice 0.83 vs SynthSeg 0.76; worst-case .59 vs .10** | PMC11661081, 2024 |
| **Chalcroft (stroke)** | **Bloch physics** from tissue maps, **params sampled from literature dist.** | brain/stroke, multi-seq | yes, synth-only | domain-agnostic stroke seg | arXiv 2412.03318, 2024 |
| **Reverse Imaging** | **inverse**: infer T1/T2/PD from bSSFP (diffusion prior) → re-synth | cardiac | aug | MYO 47→**86.5** on unseen MOLLI | arXiv 2508.21254, 2025 |
| **FASTR-SCANN** | T1 relaxation curve → synth T1w contrasts | cardiac T1-mapping | yes | automated T1/ECV seg | Radiology:AI 2022, ryai.210294 |
| **Gheorghiță (cardiac)** | **GauGAN** mask→image (not physics) | cardiac cine | pretrain+finetune | EF RMSE 7.1→3.7% | Sci Rep 2022, s41598-022-06315-3 |
| **MRXCAT** | XCAT physics phantom | cardiac | eval mostly | recon/robustness testbed | MRXCAT phantom |
| **KomaMRI / JEMRIS** | full Bloch solver | general | "expected" | ML-data aspirational | KomaMRI arXiv 2301.02702 |
| **Ours** | **bSSFP eq. from LITERATURE T1/T2/PD + SSM shapes**, swept acq | **cardiac cine** | yes, synth-only + aug | 0.56 synth / 0.85 real-multi-vendor | — |

## GAPS — what nobody seems to do (our opportunity surface)

1. **Cardiac-cine physics-label-painting from LITERATURE params + generative SSM shapes.**
   UltimateSynth (physics) is brain + needs MRF maps; cardiac physics work is inverse or GAN. Our
   exact cell — cheap (no qMRI), shape-generative (SSM), cardiac — is open.
2. **Parametric (interpretable) inverse twin.** Reverse Imaging uses a diffusion prior; a *parametric*
   fit recovering acquisition/tissue params (a genuine digital twin) is less explored.
3. **Thin-structure partial-volume realism** in synthetic training data (Q8) — under-explored;
   our myo over-spread quantification could be a contribution.
4. **The reconciled "physically-constrained diversity" framing** with a cardiac cross-vendor triad
   (real / synth-only / synth+DA) measured honestly — a clean methods story.

## HOW OUR APPROACH COMPARES (assessment)

- **Not novel in principle** — physics-constrained label-conditioned synthesis is established and
  *proven superior to random-GMM* (UltimateSynth). Our core bet is correct and backed.
- **Novel in the specific combination** — cardiac cine + closed-form bSSFP + **literature** tissue
  params (no per-subject qMRI) + **SSM shape generation** + honest cross-vendor triad. UltimateSynth's
  dependence on MRF maps is exactly the cost we avoid; the tradeoff (literature vs subject-specific
  params) is our story to tell.
- **Behind on the inverse** — Reverse Imaging (2025) published our twin idea first, with strong
  numbers, and hit the same identifiability wall. We should cite it as SOTA and differentiate on
  *parametric* fitting + cross-vendor, not claim priority.
- **Honest positioning** — no pure-synthetic method beats real-multi-vendor cross-vendor cardiac; our
  0.56-vs-0.85 and "synth = augmentation / unseen-domain / annotation-free" framing is field-accurate,
  not a weakness to hide.
- **Action items surfaced:** (a) prioritise literature-param **sampling** (bd `04bh`) — endorsed by
  Chalcroft; (b) reframe the internal note from "diversity>fidelity" to "**physically-constrained
  diversity**"; (c) treat thin-wall PV as a possible contribution; (d) cite Reverse Imaging +
  UltimateSynth as the two anchor references in any writeup.

## Sources
- SynthSeg — Billot et al., *Medical Image Analysis* 2023 — https://arxiv.org/abs/2107.09559 · https://github.com/BBillot/SynthSeg
- UltimateSynth (MRI physics for pan-contrast AI) — https://pmc.ncbi.nlm.nih.gov/articles/PMC11661081/
- Reverse Imaging — arXiv 2508.21254 (2025) — https://arxiv.org/html/2508.21254
- Chalcroft et al., physics-constrained stroke synthesis — arXiv 2412.03318 (2024) — https://arxiv.org/pdf/2412.03318
- FASTR-SCANN, relaxation-based synthetic contrast (cardiac T1) — https://pubs.rsna.org/doi/10.1148/ryai.210294
- Gheorghiță et al., synthetic cardiac cine (GauGAN) — https://www.nature.com/articles/s41598-022-06315-3 · https://pmc.ncbi.nlm.nih.gov/articles/PMC8844403/
- KomaMRI.jl — arXiv 2301.02702; *MRM* 2023 — https://pmc.ncbi.nlm.nih.gov/articles/PMC10952765/
- M&Ms challenge — Campello et al., *IEEE TMI* 2021 — https://www.semanticscholar.org/paper/edfc14cc234122533a4f3728a36e54dfd6de6211
- Domain randomization / fidelity tradeoff — https://lilianweng.github.io/posts/2019-05-05-domain-randomization/ · Benchmarking DR arXiv 2011.07112
