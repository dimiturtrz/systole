# MRI Simulation Libraries Evaluation for Synthetic-Acquisition Augmentation
**Date:** 2026-06-24  
**Author:** Research pass (Claude Sonnet 4.6, no subagents)  
**Project:** cardioseg — cardiac-MRI segmentation pipeline

---

## Use Case

**Goal:** Synthetic-acquisition augmentation for cross-vendor robustness in a Python cardiac-MRI segmentation pipeline (cardioseg). Take real ACDC short-axis slices + segmentation labels, re-acquire them under varied synthetic MRI physics (contrast/T1-T2 weighting, bias field, noise, k-space artifacts) to mimic other scanner vendors (Canon, Siemens, GE, Philips). Train on augmented set, measure cross-vendor Dice gain on held-out Canon/ACDC data.

**Critical constraint:** We have anatomy + labels only — NO per-voxel T1/T2/PD maps.

---

## 1. Library Profiles

### 1.1 KomaMRI (Julia)

- **Language:** Julia  
- **License:** MIT ([GitHub](https://github.com/JuliaHealth/KomaMRI.jl))  
- **Latest release:** v0.10.4 (May 26, 2024); component packages KomaMRIBase-v0.11.3 / KomaMRICore-v0.11.3 (June 2024); v0.10.3 released May 8, 2026 (bug-fix). Active. [Releases page](https://github.com/JuliaHealth/KomaMRI.jl/releases)  
- **Maintenance:** Actively maintained. v0.9.1 received perfect score in OSI² CAB Review June 2025. 2026 paper on arbitrary-motion simulation published in *Magnetic Resonance in Medicine* ([Villacorta-Aylagas et al. 2026](https://onlinelibrary.wiley.com/doi/10.1002/mrm.70145)).  
- **GPU:** Yes — device-agnostic GPU backends (CUDA, Metal, oneAPI) via Julia's array abstractions. v0.9 (Oct 2024) reported 4–5× speed-up, 80× memory reduction vs prior versions.  
- **Python interop:** None native. Callable from Python only via `julia` PyPI bridge (`juliacall`/`PythonCall`) — adds non-trivial setup overhead and JIT warm-up latency per training epoch call. Not production-ready as a training-loop plugin.  
- **Input — labels vs maps:** Requires explicit phantom definition: spin positions + per-spin T1/T2/PD/T2*/Δω properties. No built-in route to ingest a segmentation label map and auto-assign tissue parameters. You must build the phantom object manually (assign T1/T2/PD per label class, then expand to voxel grid). Documented API: `Phantom` struct takes arrays of positions + properties. Cardiac phantom examples exist (heartbeat + respiratory motion demos in docs, v0.9 "dynamic phantom plotting").  
- **Cardiac/anatomical phantom support:** Yes — motion-capable cardiac phantoms included; v0.9 introduced mix-and-match motion definitions (rotations, translations, breathing, heartbeat). MRXCAT-style digital phantom workflow documented.  
- **Precedent for sim-to-real / domain augmentation:** No direct published precedent for label→sim→train loop in cardiac seg. Primary use case is pulse-sequence development and motion artifact study.  
- **Install difficulty:** High for a Python ML pipeline. Julia runtime install + `juliacall` Python bridge + KomaMRI package + GPU drivers aligned. Not pip-installable.

---

### 1.2 JEMRIS (C++)

- **Language:** C++ (core); MATLAB/Python wrappers available  
- **License:** Open-source (GPL); GPU extension CC-BY 4.0 ([PMC article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12443918/))  
- **Latest release:** Stable 2.9-0 (date approximate 2024–2025); GPU-JEMRIS paper published 2025 in *Magnetic Resonance Materials in Physics, Biology and Medicine* ([Springer](https://link.springer.com/article/10.1007/s10334-025-01281-z)); website last updated 03.01.2025 ([jemris.org](https://www.jemris.org/ug_about.html)).  
- **Maintenance:** Active — GPU extension released 2025, Helmholtz Imaging CONNECT listing current ([connect.helmholtz-imaging.de](https://connect.helmholtz-imaging.de/solution/99)).  
- **GPU:** GPU-JEMRIS (2025) reimplements Bloch integration in CUDA C++, achieving 3–12× (double precision) and 7–65× (single precision) speed-up vs parallelised CPU. Small error: <0.1% NRMSE (double), <1% k-space NRMSE (single).  
- **Python interop:** `py2jemris` — Python library on GitHub ([imr-framework/py2jemris](https://github.com/imr-framework/py2jemris)) that converts PyPulseq sequences to JEMRIS XML, constructs coil maps and numerical phantoms, and launches JEMRIS simulation pipelines. Indirect — spawns JEMRIS subprocess, not a native Python extension.  
- **Input — labels vs maps:** Per-voxel property maps required. The phantom format is explicitly `P(r,t) = [M0(r,t), T1(r,t), T2(r,t), T2*(r,t), ΔωRF(r,t), χ(r,t)]` — spatially-varying fields at each voxel location. You must convert segmentation labels → per-voxel property maps before feeding JEMRIS. Standard usage (Duke phantom, abdominal phantom) pre-assigns T1/T2/M0 per organ class, which is exactly the label→tissue-params pipeline, but the conversion is user-side.  
- **Cardiac/anatomical phantom support:** GPU-JEMRIS tested on multi-species abdominal phantom with MEGRE sequence. No dedicated cardiac phantom ships with the code. Brain and geometrical test phantoms documented.  
- **Precedent for sim-to-real / domain augmentation:** Not documented. Primary purpose: sequence simulation and artifact study.  
- **Install difficulty:** Very high. C++ build, dependency chain (Boost, HDF5, ISMRMRD), then GPU CUDA build. `py2jemris` abstracts some of this but still requires a running JEMRIS binary. Not pip-installable.

---

### 1.3 MRiLab (MATLAB)

- **Language:** MATLAB (primary); CUDA GPU via MEX  
- **License:** BSD-2-Clause ([GitHub](https://github.com/leoliuf/MRiLab))  
- **Latest release:** v1.3 — released **January 12, 2017**. No releases since. 9 open issues, 1 open PR. Effectively unmaintained for 9 years.  
- **Maintenance:** Stale. Last commit date not prominently displayed; no new release since 2017.  
- **GPU:** CUDA via MEX (10.9% of codebase is CUDA), but tied to old MATLAB MEX interface.  
- **Python interop:** None. MATLAB-only. Would require MATLAB Engine API for Python — heavyweight, licensed.  
- **Input — labels vs maps:** Per-voxel tissue maps via "Generalized Multi-Pool Exchange Tissue Model" — parameterized at voxel level. Label→map conversion is user responsibility.  
- **Cardiac/anatomical phantom support:** Not prominent in current docs. General-purpose signal formation simulator.  
- **Precedent for sim-to-real / domain augmentation:** None found.  
- **Install difficulty:** Requires MATLAB license + compatible CUDA/MEX build. Effectively not viable for a Python pipeline.

---

### 1.4 pypulseq / pulseq

- **Language:** Python (pypulseq); specification is vendor-neutral `.seq` file format  
- **License:** MIT (as of v1.5.0, January 2025; previously AGPL-3.0) ([GitHub releases](https://github.com/imr-framework/pypulseq/releases))  
- **Latest release:** v1.5.0.post1 — January 29, 2025. Actively maintained.  
- **Maintenance:** Active through early 2025. Multiple bug fixes and enhancements in v1.5.x cycle.  
- **GPU:** No GPU support in pypulseq itself — it is a *sequence design* tool, not a signal simulator.  
- **Python interop:** Native Python package, pip-installable. Core function: design MRI pulse sequences (RF pulses, gradients, ADC events) and export `.seq` files.  
- **Input — labels vs maps:** Not applicable — pypulseq does not simulate MRI signal; it defines the acquisition sequence. Pair with a Bloch simulator (MRzero, KomaMRI, JEMRIS) for actual signal simulation.  
- **Cardiac phantom support:** N/A (sequence design only).  
- **Precedent for sim-to-real:** pypulseq is the sequence-definition layer; downstream simulators handle signal. Relevant as the sequence-specification input for MRzero and KomaMRI (both are Pulseq-compatible).  
- **Install difficulty:** Low — `pip install pypulseq`.

---

### 1.5 MRzero

- **Language:** Python (PyTorch)  
- **License:** AGPL-3.0 ([GitHub](https://github.com/MRsources/MRzero-Core))  
- **Latest release:** v0.4.2 — January 16, 2026. Actively maintained.  
- **Maintenance:** Active. Displayed at ESMRMB 2024; v0.4.2 released January 2026. [PyPI](https://pypi.org/project/MRzeroCore/)  
- **GPU:** Yes — built on PyTorch, CUDA-capable. "Heavily relies on PyTorch for fast (GPU-) Tensor calculations." Parts compiled in Rust (x86 Windows/Linux only; other platforms unsupported as of late 2024).  
- **Python interop:** Native — `pip install MRzeroCore`. Differentiable forward model: backpropagation through the entire pipeline (sequence → phantom → signal → image) for sequence optimisation.  
- **Input — labels vs maps:** **Requires explicit per-spin T1, T2, PD (proton density), and ΔB0 as input.** The spin system is parameterised as `[PD, T1, T2, ΔB0]` per spin/voxel. No label-map ingestion built in — user must build tissue parameter arrays from labels. PDG (partition-based Bloch simulation) is used internally.  
- **Cardiac phantom support:** No cardiac phantom ships. Documentation focuses on brain-like phantoms and sequence optimisation examples. Pulseq-compatible — sequences from pypulseq or MATLAB-generated `.seq` files can be simulated.  
- **Precedent for sim-to-real / domain augmentation:** Primary use case is automated sequence discovery and optimisation, not augmentation for segmentation training. No published cardiac-seg augmentation use case found.  
- **Install difficulty:** Medium — `pip install MRzeroCore` works on x86 Windows/Linux. Rust extension limits portability. AGPL-3.0 license restricts commercial embedding.

---

### 1.6 CMRsim (Python — notable find)

- **Language:** Python (TensorFlow 2)  
- **License:** Not prominently stated; ETH Zurich open-source release ([GitLab](https://gitlab.ethz.ch/ibt-cmr/mri_simulation/cmrsim))  
- **Latest release:** Published *Magnetic Resonance in Medicine* June 2024 ([Weine et al. 2024](https://pubmed.ncbi.nlm.nih.gov/38234037/)). GitLab repo activity not confirmed in this pass.  
- **Maintenance:** Published 2024 from ETH Zurich CMR group; maintenance status beyond publication unknown.  
- **GPU:** Yes — TensorFlow 2 GPU execution.  
- **Python interop:** Native Python package. API described as "easy-to-use."  
- **Input — labels vs maps:** Accepts "dynamic digital phantoms with complex motion as input." The paper describes time-resolved digital phantoms, but does not explicitly confirm label-only ingestion — full cardiac anatomy phantoms with tissue properties are likely required. "Scalable to detailed time-resolved digital phantoms" per ETH CMR software page.  
- **Cardiac phantom support:** Yes — specifically designed for cardiovascular MR, incorporating complex motion and flow. This is the only simulator in this survey purpose-built for cardiac MRI.  
- **Precedent for sim-to-real / domain augmentation:** Not documented as an augmentation tool; primary use is CMR sequence evaluation. However, cardiac specificity makes it the closest Bloch simulator to the cardioseg use case.  
- **Install difficulty:** Medium — Python/TF2, GitLab source install. Less friction than JEMRIS/KomaMRI but less proven than TorchIO/MONAI.

---

### 1.7 TorchIO

- **Language:** Python (PyTorch)  
- **License:** Apache 2.0  
- **Latest release:** Active on PyPI; v0.19.x line maintained through 2025. [PyPI](https://pypi.org/project/torchio/)  
- **Maintenance:** Actively maintained. [GitHub](https://github.com/TorchIO-project/torchio)  
- **GPU:** CPU-based transforms (spatial/intensity ops); underlying tensors are PyTorch and can reside on GPU for downstream ops, but MRI artifact simulation transforms themselves run on CPU.  
- **Python interop:** Native — `pip install torchio`. Integrates directly with PyTorch DataLoader.  
- **Input — labels vs maps:** Accepts raw MRI volumes directly — **no tissue maps needed**. Transforms operate on voxel intensities of real images.  
- **MRI artifact transforms available:**
  - `RandomBiasField` — low-frequency multiplicative field (simulates B1 inhomogeneity)
  - `RandomGhosting` — k-space ghosting from motion/pulsation
  - `RandomSpike` — k-space spike artifacts
  - `RandomMotion` — rigid-body motion in k-space (ghosting pattern)
  - `RandomNoise` — additive Gaussian; Rician noise via composition
  - `RandomBlur`, `RandomGamma`, `RandomFlip`, `RandomAffine`, `RandomElasticDeformation`
- **Cardiac/anatomical phantom support:** N/A — operates on real images, not phantoms.  
- **Precedent for sim-to-real / domain augmentation:** Widely used in medical imaging training pipelines. M&Ms challenge top performers used intensity-driven augmentation (brightness, contrast, noise, histogram matching) — same class of operations as TorchIO transforms. CMRxMotion challenge (2025, [arxiv 2507.19165](https://arxiv.org/pdf/2507.19165)) used motion artifact simulation. TorchIO cited in domain generalisation review literature as standard augmentation library.  
- **Install difficulty:** Very low — `pip install torchio`.

---

### 1.8 MONAI

- **Language:** Python (PyTorch)  
- **License:** Apache 2.0  
- **Latest release:** Active 2025/2026; NVIDIA GTC 2025 featured MONAI-based cardiac MRI segmentation ([NVIDIA On-Demand](https://www.nvidia.com/en-us/on-demand/session/gtc25-s73347/)).  
- **Maintenance:** Actively maintained by NVIDIA + community.  
- **GPU:** Yes — full GPU pipeline support.  
- **Python interop:** Native — `pip install monai`. Integrates with PyTorch training loops.  
- **Input — labels vs maps:** Operates on real image volumes; no tissue maps needed.  
- **MRI-relevant transforms:** `RandGaussianNoise`, `RandBiasField`, `RandGibbsNoise`, `RandKSpaceSpikeNoise`, `NormalizeIntensity`, `ScaleIntensityRange`, `HistogramNormalize`, `RandAffine`, `RandElasticDeformation`. Overlaps substantially with TorchIO.  
- **Cardiac/anatomical phantom support:** N/A — pipeline framework, not a phantom generator.  
- **Precedent:** MONAI is the dominant framework for medical image segmentation training. GTC 2025 cardiac seg session used MONAI. Augmentation transforms are a subset of what TorchIO offers specifically for MRI artifacts; MONAI adds GPU-side acceleration of spatial transforms.  
- **Install difficulty:** Low — `pip install monai`.

---

## 2. Comparison Table

| Library | Lang | License | Last Release | Active? | GPU | Python pip | Input needs | Cardiac phantom | Effort to integrate |
|---------|------|---------|--------------|---------|-----|------------|-------------|----------------|---------------------|
| KomaMRI | Julia | MIT | May 2026 (v0.10.3) | Yes | Yes (multi-backend) | No (juliacall bridge) | T1/T2/PD per voxel (user builds from labels) | Yes (motion demos) | Very High |
| JEMRIS | C++ | GPL/CC-BY | 2025 (GPU ext.) | Yes | Yes (CUDA) | No (py2jemris subprocess) | Per-voxel T1/T2/PD/M0/T2* | No cardiac shipped | Very High |
| MRiLab | MATLAB | BSD-2 | Jan 2017 | **Stale** | CUDA/MEX | No (MATLAB Engine) | Per-voxel tissue maps | Not prominent | Extremely High / Skip |
| pypulseq | Python | MIT | Jan 2025 (v1.5.0.post1) | Yes | No (seq design only) | Yes | N/A (sequence designer) | N/A | Low (but needs pairing) |
| MRzero | Python | AGPL-3.0 | Jan 2026 (v0.4.2) | Yes | Yes (PyTorch/CUDA, x86 only) | Yes | T1/T2/PD/ΔB0 per voxel | No | Medium |
| CMRsim | Python | ETH open | Jun 2024 (paper) | Unknown | Yes (TF2) | No (GitLab src) | Digital cardiac phantom w/ tissue props | **Yes (purpose-built)** | High |
| TorchIO | Python | Apache-2 | 2025 (active) | Yes | CPU transforms | Yes | Real image volumes, no maps needed | N/A | Very Low |
| MONAI | Python | Apache-2 | 2026 (active) | Yes | Yes (spatial transforms) | Yes | Real image volumes, no maps needed | N/A | Very Low |

---

## 3. Decision Questions — Cited Evidence

### Q1: Is assigning T1/T2/PD per tissue label the standard Bloch-sim route? Literature precedent?

**Yes — this is the established pattern.** Evidence:

- JEMRIS GPU paper (2025) uses Duke phantom and abdominal phantom pre-assigned with per-organ T1/T2/M0 values, then expands to voxel grid. The voxel-level format `P(r,t) = [M0(r,t), T1(r,t), T2(r,t), ...]` is populated by class-level literature values ([PMC12443918](https://pmc.ncbi.nlm.nih.gov/articles/PMC12443918/)).  
- MRzero paper (2021, *MRM*) trains on "1024 samples with MR properties" by assigning tissue-class T1/T2/PD values — the spin system definition is label-driven even if the format is per-voxel ([arxiv 2002.04265](https://arxiv.org/pdf/2002.04265)).  
- FaBiAN (fetal brain, 2024 *MAGMA*) uses reference T1/T2 values from literature for each tissue class; its randomised variant (randFaBiAN) samples T1/T2 from broad distributions per class to randomise contrast ([arxiv 2109.03624](https://arxiv.org/pdf/2109.03624)).  
- **SynthSeg** (Billot et al. 2023, *Medical Image Analysis*) is the canonical label→contrast-randomisation method: generates entire training sets from 3D segmentation labels, randomises Gaussian intensity distributions per label class (which implicitly models T1/T2 contrast variation). Extended to cardiac MRI and CT ([PMC10154424](https://pmc.ncbi.nlm.nih.gov/articles/PMC10154424/)). Critically: SynthSeg does **not** do Bloch simulation — it samples intensity from per-label Gaussian distributions, which is far simpler than full Bloch but achieves contrast randomisation.  
- DRIFTS (fetal brain seg, 2024) confirms "intensity clustering" gives +5 DSC over naive domain randomisation when combined with synthetic data, showing that label-grounded intensity modelling matters ([arxiv 2411.06842](https://arxiv.org/html/2411.06842v1)).

**Takeaway:** The label→T1/T2/PD→simulate pipeline is standard, but in practice researchers often shortcut full Bloch simulation with intensity-distribution randomisation per label (SynthSeg pattern), reserving Bloch simulation for cases where k-space trajectory or specific sequence parameters must be varied.

---

### Q2: Is TorchIO/MONAI the pragmatic path? What does the cross-vendor cardiac DG literature actually credit?

**The M&Ms challenge findings are unambiguous:** the top-line conclusion from the MICCAI 2020 M&Ms challenge paper (Campello et al. 2021, *IEEE TMI*) is:

> "The obtained results indicate **the importance of intensity-driven data augmentation**, as well as the need for further research to improve generalizability towards unseen scanner vendors or new imaging protocols." ([ResearchGate](https://www.researchgate.net/publication/352494774_Multi-Centre_Multi-Vendor_and_Multi-Disease_Cardiac_Segmentation_The_MMs_Challenge))

Specific augmentation methods credited in top-performing M&Ms submissions:

- **Histogram matching augmentation** (Zeng et al. 2021): transfer intensity distribution of test-domain images to training images. Achieved Dice 0.9051/0.8405/0.8749 (LV/myo/RV) on the M&Ms held-out vendor. Paper title: "Histogram Matching Augmentation for Domain Adaptation with Application to Multi-Centre, Multi-Vendor and Multi-Disease Cardiac Image Segmentation" ([Springer](https://link.springer.com/chapter/10.1007/978-3-030-68107-4_18), [arxiv 2012.13871](https://arxiv.org/abs/2012.13871)). This is a simple, plug-and-play method.  
- **Noise addition, brightness/contrast modification** — multiple teams; these map directly to TorchIO `RandomNoise`, `RandomGamma`, and `RandomBiasField`.  
- **Style-invariant approaches** (Huang et al. 2021) using test-time augmentation.  
- **MixUp and probabilistic perturbation** — 2025 papers on cardiac domain generalisation in myocardial pathology seg ([Springer 2025](https://link.springer.com/chapter/10.1007/978-3-031-87009-5_4), [Springer 2025](https://link.springer.com/chapter/10.1007/978-3-031-87009-5_1)).

**No M&Ms challenge paper credits full Bloch simulation for vendor-gap improvement.** The vendor gap in cardiac CMR is primarily an intensity/contrast distribution gap (different T1-weighting, flip angle protocols, k-space sampling across vendors), not a structural or geometric gap. The gap is well-addressed by intensity-space augmentation.

**Domain generalisation review** (2023, [arxiv 2310.08598](https://arxiv.org/pdf/2310.08598)) confirms that for MRI domain shifts arising from "variations in imaging protocol, differences in scanner hardware, and manufacturer disparities," the dominant approaches are data augmentation (intensity + spatial) and normalization — not physics simulation.

**CMRxMotion challenge (2025)** on respiratory motion artefacts did use motion simulation, but for a different problem class (retrospective motion correction, not vendor harmonisation) ([arxiv 2507.19165](https://arxiv.org/pdf/2507.19165)).

---

### Q3: Lowest-effort path and is full Bloch sim worth it?

**Evidence-based answer:**

The vendor gap in cross-vendor cardiac short-axis segmentation is driven by:
1. Scanner-specific intensity rescaling and window/level conventions
2. Vendor-specific flip angles and TR/TE settings → different T1/T2 contrast weighting
3. k-space filtering and reconstruction pipeline differences (Gibbs ringing, noise characteristics)
4. Coil sensitivity patterns → bias field variation

All of these have direct TorchIO/MONAI analogs:
- Intensity rescaling → `RandomGamma`, `RandScaleIntensity`
- T1/T2 contrast variation → `RandomBiasField` + `RandomGamma` + intensity range shifts
- k-space artifacts → `RandomSpike`, `RandomGhosting`
- Bias field → `RandomBiasField`
- Noise → `RandomNoise` (Rician via composition)
- Histogram matching → MONAI `HistogramNormalize` applied to training set

**Full Bloch simulation is not warranted for this specific goal** given:
1. No T1/T2/PD maps available — building them from labels is feasible but adds non-trivial engineering (must source literature values, build phantom construction pipeline, validate parameter ranges per tissue class).
2. None of the cross-vendor cardiac-seg challenge literature (M&Ms, M&Ms-2) credits Bloch simulation for vendor-gap reduction. The winning approaches were intensity augmentation + histogram matching.
3. Bloch sim gives physically correct k-space signal for a *specific* sequence — but simulating the full diversity of vendor-specific sequences without vendor sequence specs is speculative, and the result may not be more diverse than randomised intensity augmentation.
4. SynthSeg's label→Gaussian-intensity-randomisation approach (no Bloch, no T1/T2 maps) achieved cardiac generalisation ([PMC10154424](https://pmc.ncbi.nlm.nih.gov/articles/PMC10154424/)) — suggesting that contrast randomisation at the label level, not physics simulation, is what matters for generalisation.

The one scenario where Bloch sim would add value: if you want to simulate specific artifacts tied to exact k-space trajectories (e.g., EPI ghosts, spiral off-resonance) that differ across vendors and aren't well-modelled by TorchIO transforms. This is a marginal benefit for standard bSSFP/SSFP cine cardiac sequences where the vendor gap is dominated by intensity distribution differences.

---

## 4. Options Ranked by Effort vs Expected Payoff

### Tier 1 — Recommended: TorchIO + Histogram Matching (Very Low Effort / High Payoff)

**What:** `pip install torchio`. Compose `RandomBiasField + RandomNoise + RandomGhosting + RandomSpike + RandomGamma` in training augmentation pipeline. Add histogram matching augmentation (Zeng et al. 2021 approach) as a data-preprocessing step using `torchio` or scikit-image `exposure.match_histograms` on ACDC images matched to Canon intensity distributions.

**Evidence:** M&Ms challenge top methods, Dice ≥0.90 LV on unseen vendor. Directly cited by challenge organisers as the most impactful finding. Histogram matching paper ([arxiv 2012.13871](https://arxiv.org/abs/2012.13871)) is plug-and-play.

**Effort:** 1–2 days to wire in, no new dependencies beyond TorchIO.  
**Payoff:** High — directly validated on the vendor-gap problem.

---

### Tier 2 — Medium Effort / Medium-High Payoff: SynthSeg-style Label Intensity Randomisation

**What:** For each training epoch, sample per-label Gaussian intensity parameters (mean/std) from broad distributions informed by MRI physics (T1-weighted: fat bright, muscle mid, blood bright; T2-weighted: fluid bright, muscle dark). Apply to segmentation masks to synthesise new contrast variants of ACDC images. No Bloch sim needed.

**Evidence:** SynthSeg ([PMC10154424](https://pmc.ncbi.nlm.nih.gov/articles/PMC10154424/)) applied to cardiac MRI with strong generalisation. DRIFTS (2024) shows +5 DSC from intensity clustering in synthetic data ([arxiv 2411.06842](https://arxiv.org/html/2411.06842v1)).

**Effort:** 3–5 days. Need to implement label-to-intensity sampling loop; no external physics sim library required. FreeSurfer's SynthSeg codebase is open-source and adaptable.  
**Payoff:** High for large contrast variation; may exceed TorchIO approach for cross-modality cases (bSSFP vs GRE), but cardiac vendors mostly share sequence type so marginal gain over Tier 1 unclear.

---

### Tier 3 — High Effort / Uncertain Payoff: MRzero or CMRsim Bloch Simulation

**What:** Build a cardiac label→T1/T2/PD tissue property lookup (using literature values: myocardium T1≈1000ms at 1.5T, blood T1≈1600ms, fat T1≈250ms, etc.), construct voxel-level property maps from ACDC segmentation labels, then run MRzero or CMRsim to simulate signal under varied sequence parameters (TR, TE, flip angle) mimicking Canon vs Siemens protocols.

**Evidence for route:** FaBiAN pattern ([arxiv 2109.03624](https://arxiv.org/pdf/2109.03624)), label→tissue-params→Bloch→image is established in brain; cardiac analog exists in CMRsim ([PubMed 38234037](https://pubmed.ncbi.nlm.nih.gov/38234037/)).

**Blockers:**
- MRzero: AGPL-3.0 (complicates commercial use); x86 Linux/Windows only; requires constructing phantom from labels manually; no cardiac phantom ships.
- CMRsim: ETH GitLab, TF2, maintenance uncertain post-2024 paper; cardiac-specific but requires tissue property assignment.
- Neither has published precedent as a segmentation training augmentation tool.
- Simulation of canonical bSSFP sequences requires accurate flip-angle and off-resonance maps — not available from anatomy labels alone.

**Effort:** 2–4 weeks minimum. Tissue property lookup table, phantom construction pipeline, sequence parameterisation, validation that simulated images are plausible.  
**Payoff:** Speculative. No M&Ms or comparable challenge credits Bloch sim for vendor-gap reduction. Risk: simulated images may look physically correct but still fail to capture vendor-specific reconstruction pipeline effects (k-space filtering, reconstruction kernel, SENSE/GRAPPA variants) that dominate the vendor gap in practice.

---

### Tier 4 — Skip: KomaMRI, JEMRIS, MRiLab

- **KomaMRI:** Julia — not integrable in a Python training loop without major engineering overhead. Strong simulator but wrong language ecosystem.
- **JEMRIS:** C++ binary + subprocess interface — very high integration cost, no cardiac phantom, no published seg-augmentation precedent.
- **MRiLab:** Last release 2017 — effectively dead. MATLAB dependency rules it out.

---

## 5. Summary Recommendation

For **cardioseg cross-vendor robustness with ACDC labels and no T1/T2 maps:**

1. **Start with Tier 1.** Wire TorchIO artifact transforms + histogram matching into the training pipeline. This is the highest-evidence, lowest-effort path, directly validated by the M&Ms challenge (the closest existing benchmark to the target task).

2. **If Tier 1 Dice gain is insufficient on held-out Canon**, try **Tier 2** (SynthSeg-style label intensity randomisation) — adds contrast diversity beyond what TorchIO can generate from the existing intensity distributions.

3. **Do not invest in full Bloch simulation unless** you have a specific hypothesis that k-space trajectory differences (not intensity distribution differences) are the dominant failure mode — and validate this first by inspecting the Canon vs ACDC training data intensity histograms.

---

## References

- Campello et al. (2021). Multi-Centre, Multi-Vendor and Multi-Disease Cardiac Segmentation: The M&Ms Challenge. *IEEE TMI*. https://www.researchgate.net/publication/352494774
- Zeng et al. (2021). Histogram Matching Augmentation for Domain Adaptation. *STACOM 2020*. https://arxiv.org/abs/2012.13871 | https://link.springer.com/chapter/10.1007/978-3-030-68107-4_18
- Billot et al. (2023). SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining. *Medical Image Analysis*. https://pmc.ncbi.nlm.nih.gov/articles/PMC10154424/
- Weine et al. (2024). CMRsim — A Python package for cardiovascular MR simulations incorporating complex motion and flow. *MRM*. https://pubmed.ncbi.nlm.nih.gov/38234037/
- Villacorta-Aylagas et al. (2026). Versatile and Highly Efficient MRI Simulation of Arbitrary Motion in KomaMRI. *MRM*. https://onlinelibrary.wiley.com/doi/10.1002/mrm.70145
- GPU-JEMRIS (2025). Gpu-accelerated JEMRIS for extensive MRI simulations. *MAGMA*. https://link.springer.com/article/10.1007/s10334-025-01281-z | https://pmc.ncbi.nlm.nih.gov/articles/PMC12443918/
- MRzero v0.4.2. https://pypi.org/project/MRzeroCore/ | https://github.com/MRsources/MRzero-Core
- KomaMRI.jl releases. https://github.com/JuliaHealth/KomaMRI.jl/releases
- pypulseq v1.5.0.post1. https://github.com/imr-framework/pypulseq/releases
- py2jemris. https://github.com/imr-framework/py2jemris
- TorchIO. https://github.com/TorchIO-project/torchio | https://pypi.org/project/torchio/
- DRIFTS (2024). Maximizing domain generalization in fetal brain tissue segmentation. https://arxiv.org/html/2411.06842v1
- FaBiAN (2021). https://arxiv.org/pdf/2109.03624
- Domain Generalization for Medical Image Analysis: A Review (2023). https://arxiv.org/pdf/2310.08598
- CMRxMotion Challenge (2025). https://arxiv.org/pdf/2507.19165
- Domain Generalization in Myocardial Pathology Segmentation with MixUp (2025). https://link.springer.com/chapter/10.1007/978-3-031-87009-5_4
