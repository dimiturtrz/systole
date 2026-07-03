# Generation architecture — one process, many operations

> Status: **living first cut** (2026-07-02). The point is to give the tangle of synth pipelines a
> shared skeleton so we can fix it piece by piece. Expect this to evolve — emergent architecture.

## The one idea

There is **one generative process** — a recipe that turns hidden causes into an image:

```
anatomy  →  tissue properties  →  acquisition (scanner)  →  clean image  →  framing + noise  →  observed image
(shape)     (T1/T2/PD)            (field, TR, flip)                          (pan/rotate/FOV/noise)
```

Every "pipeline" we have is **not a separate system** — it's a different way of *using this one
process*. Once you see that, the clusterfuck becomes one labelled graph with a few operations on it.

This is the classic **analysis-by-synthesis** idea: *model how images are made, then run the model
forwards to generate and backwards to analyse.* The graph above is a **generative process** (a.k.a. a
structural/causal model): each arrow is a mechanism, each box is a factor that causes the next.

## The factors (the hidden causes), grouped

We call the full set of hidden causes **θ** (theta). It splits into three groups:

| group | what | where in code |
|---|---|---|
| **shape** (anatomy) | the label map: which voxel is RV-cav / myo / LV-cav / bg | `anatomy.voxelize` (SSM meshes), or a real GT mask |
| **color** (appearance) | tissue T1/T2/PD **and** acquisition (field, TR, flip) → contrast | `mri_physics` + `Acquisition` strategy in `synth.py` |
| **nuisance** (framing) | position/pan, rotation, field-of-view margin, blur, noise | `augment.py` + the corruption chain in `synth.py` |

Each group can be sourced three ways — this is the **control axis**:

- **real / given** — take it from real data (e.g. real masks for shape)
- **parametric** — set interpretable physics knobs (SSM weights, bSSFP params) ← *where we are*
- **learned** — sample from a learned prior (VAE/diffusion) ← *later, and carefully (see steering)*

## The three operations (this is the whole API)

Everything we do is one of three operations on the process. (The fancy names are in the glossary.)

1. **SAMPLE** — pick θ from a prior, run forwards → a new image. *("generate")*
2. **FIT** — given a real image, find the θ that reproduces it (run backwards). *("invert / reconstruct")*
3. **FIX** — hold part of θ to a real value, sample the rest. *("condition")*

Our pipelines are just these operations with different factors sourced differently:

| pipeline (bead) | operation | shape | color | what it is |
|---|---|---|---|---|
| **repaint** | FIX shape=real, SAMPLE color | real mask | randomized | recolor real anatomy (**color-axis only**, flattered) |
| **generate** (`b6tb`, current full-synth) | SAMPLE shape + color | SSM (Rodero) | randomized | full generation (shape+color) |
| **augmentation** (`pwih`) | SAMPLE synth, MIX with real | mixed | randomized | synth as a robustness additive to real training |
| **inverse / digital twin** (`ncph`) | FIT | real (given) | **controlled** (fit) | recover interpretable qMRI params from a scan |
| **harmonization** (future) | FIT then re-SAMPLE color | real | re-rendered | "same heart, other scanner" (a counterfactual) |
| **torso** (`hpy`) | richer shape+bg factor | whole-FOV | randomized | structured other-organs instead of blob bg |
| **coverage metric** (`uy4d`) | measure SAMPLE vs real | — | — | does generated θ-space **encompass** real |

## The honest reporting rule (state scope upfront)

Because shape and color are separate factors, **always say which is real and which is generated**:

- **repaint** = real shape + generated color → measures **color** generalization *alone* (~0.68 xval).
- **generate** = generated shape + generated color → **full** generation (~0.56 xval).
- `0.68 − 0.56 ≈ 0.12` = the **isolated cost of generating shape** (color held constant).

Neither is cheating; they measure different factors. Never present a repaint number as "generation".

## Per-factor metrics (don't collapse them)

Each factor needs its **own** coverage check — one global number hides which factor is failing:

- **shape** coverage → spatial **embedding** overlay: does generated shape-space cover real? (`uy4d`).
  *(Per-class intensity W1 is anatomy-agnostic — it CANNOT see the shape gap. Don't use it for shape.)*
- **color** coverage → per-class **W1** (`analysis/synth_fidelity.py`), split into *location* (mean shift =
  a possible systematic bug, e.g. blood over-brightness) and *shape-of-distribution* (spread; being
  **wider** than real is good coverage, not error — do NOT shrink it toward real).
- **framing** coverage → FOV margin / how far pan can go without truncation.
- **downstream** → cross-vendor Dice (the training objective).
- **inverse** → reconstruction error of FIT (the twin's objective).

Note the two objectives pull opposite ways, and that's fine: **forward/training wants wide coverage —
physically-constrained diversity (random contrast < physics < a swept physical manifold; random loses
to physics per UltimateSynth>SynthSeg, and a single fidelity point loses to the sweep); inverse/twin
wants tight fidelity.** Same process, opposite operations.

## Steering principle (how to grow this without wrecking it)

The **control axis** hides a coupling worth deciding deliberately:

- **Parametric physics mechanisms** → interpretable θ → **invertible** → you get the digital twin
  (`ncph`), harmonization, *and* training synth, all from one engine.
- **Black-box learned image generator** (pixel GAN/diffusion) → no interpretable θ → **not cleanly
  invertible** → you gain coverage but **lose the twin and interpretability** (the differentiated
  artifact). "Just generate" *spends* the inverse product — that's the hidden cost.

**Resolution:** keep the mechanisms physical/interpretable. Where coverage fails, **learn a richer
PRIOR over a factor** (e.g. `p(θ_shape)` beyond linear SSM), *not* a black-box leaf. Coverage improves,
θ stays interpretable, abduction (the twin) still works. And since the measured gap is in **shape**
(0.68→0.56), the first learned prior to invest in is the **anatomy** factor — not a pixel model.

## The framing / FOV axis (recently surfaced)

Voxelizing tight-to-bounds leaves no room for the pan/rotate nuisance factor → panning truncates or
zero-pads. **Generate a larger canvas than the model FOV** (heart smaller in frame, headroom around it)
so the nuisance transforms have plausible support. This is a *distinct* generalization axis from shape
and color — the **support of the observation**. (A `margin` in `voxelize`/`fit_square`.) Truncation
robustness ≠ position robustness; worth having even though translation-*aug* measured neutral (the
U-Net is convolution = translation-equivariant, so absolute position barely matters, but a heart
sliced by the frame edge is a genuinely different, harder input).

## Glossary (the new words)

- **analysis-by-synthesis** — model how data is generated, then invert the model to interpret data.
- **generative process / structural (causal) model** — the DAG of "cause → effect" mechanisms that
  produce an image; each arrow is a mechanism you can inspect or replace.
- **factor / latent (θ)** — a hidden cause (anatomy, tissue, acquisition, nuisance) fed into the process.
- **ancestral sampling** — SAMPLE: draw each factor top-down along the DAG, run forwards.
- **intervention `do(θ)`** — force a factor to a chosen value (= "controlled / parameterized" generation).
- **abduction** — FIT: infer the hidden factors that best explain an observed image (the inverse / twin).
- **counterfactual** — change one factor, keep the rest ("same heart, different scanner" = harmonization).
- **simulation-based inference (SBI)** — fitting a simulator's parameters to data (what FIT/`ncph` is).
- **domain randomization / sim2real** — train on widely-varied synth so a model transfers to real; the
  reason the forward path prefers **diversity over fidelity**.
- **nuisance factor** — a cause we don't care to estimate but must model (pan, rotation, noise); lives at
  the leaf of the DAG, which is why augmentations are "last stage".
- **support / manifold** — the region of possible inputs the generator can produce; "how much space we
  model". Coverage = does that region contain the real data's region.

## Generation SOURCES — a composite all-synthetic dataset (2026-07-02)

The goal is **all-synthetic training data**, and no single generator reaches the whole real manifold.
So think of it as a **portfolio of sources**, each ENTERING the DAG at a different point with a different
**control degree**, all feeding the shared painter/corruption tail and **composed into one dataset**.
Each source covers a different region; the union covers more than any one.

| source | enters at | control | anatomy coverage | status |
|---|---|---|---|---|
| **fully parametric** | top (invent shape params → mesh → paint) | HIGH — every factor a knob | limited (needs a shape generator) | painter ✓, shape-invent ✗ |
| **SSM (Rodero)** | shape = pre-built mesh → voxelize → paint | MED — SSM mode weights | healthy + MILD pathology (±3SD) | ✓ (pool_1000) |
| **label-space edit** | shape = deform existing label maps | MED — deform params | variations; pathology via dilation | build (`vpn5`) |
| **MRXCAT** | whole thing (torso+heart+physics) → we consume its **label `.vti`**, paint with our engine | LOW–MED — pathology knobs | whole-FOV + pathology + structured bg | `hpy` — adapter `core/data/dynamic/mrxcat.py` ✓ (`to_canonical`/`load_vti_labels`/`build_pool`; remap geometrically verified — myo ring encloses LV-cav). Tool vendored `external/mrxcat2` (gitignored, MIT); XCAT torso Duke-gated but example bundled. NB MRXCAT paints myo UNIFORM by construction (`fixLVTexture` meanLV) — confirms our myo over-spread is low-res-PV, not physics |
| **learned prior** | shape = sample a learned model | LOW — latent | the real manifold incl pathology | future (`vpn5` option) |

**Composition is cheap** (union of label pools → the painter is shared): `pool_composite = concat(sourceA,
sourceB, …)`. The VALUE is *diverse sources*, not one bigger source. Coverage is measured per source and
on the union (`shape_coverage`, `static_compare`), so we can see which source fills which gap — e.g. SSM
covers normals (NOR 0.49), the DCM/RV tail needs label-space or learned or MRXCAT. Control degree is a
*feature*: high-control sources (parametric) for targeted gaps, whole-thing sources (MRXCAT) for breadth.

## Current implementation map (what's built, where) — 2026-07-02

**Factors / mechanisms (the forward process):**

| DAG element | source | code | status |
|---|---|---|---|
| shape — real | real GT masks | `core/data/static/store.py` (load), `splits` | ✅ |
| shape — parametric | SSM meshes → label maps | `core/data/dynamic/anatomy.py`: `voxelize`, `_sax_align`, `_scale_to_target`, `build_pool`/`load_pool`. Meshes are a DIFFERENT data type from MRI images → live under `<data>/volumetric/meshes/{raw,processed}` (raw = Rodero `.vtu`; pools = `processed/rodero_anatomy/*.npz`), NOT under `mri/`. `build_pool` discovers by stem, prefers `.vtu` (ASCII `.vtk` pruned as a redundant dup) | ✅ |
| shape — learned | — | — | ⛔ (`vpn5`) |
| color — tissue | T1/T2/PD table | `mri_physics.py`: `TISSUE`, `tissue_params`, `_HEART`, `blood_classes` | ✅ |
| color — acquisition | field/TR/flip strategy | `synth.py`: `Acquisition` ABC → `LegacyAcq`/`RandomizedAcq`/`MatchedAcq`, `make_acquisition` | ✅ |
| color — per-vendor | vendor→its acq | `mri_physics.acquisition_for` exists but not wired into paint | 🟡 (`ex1`) |
| paint mechanism | bSSFP signal | `synth.py`: `synthesize_from_labels` (`bssfp_signal`, inflow, `banding`) | ✅ |
| background factor | whole-FOV fill | `synth.py`: `Background` ABC → `Flat`/`Procedural`/`Partition`/`HybridBg`, `make_background` | ✅ |
| nuisance — geometry | pan/rot/scale/flip | `augment.py`: `augment_batch` (translate opt-in, default 0) | ✅ (pan opt-in) |
| nuisance — corruption | pv/bias/blur/kspace/noise | `synth.py` corruption chain | ✅ |
| framing / FOV margin | canvas > FOV | — | ⛔ (`x8ne`) |
| real-bg cleanup | excise real heart | `synth.py`: `excise_heart` | ✅ (`mirs` fixed) |

**Operators:**

| operator | code | status |
|---|---|---|
| **SAMPLE** (generate) | `generator.py`: `Generator.batch` + `synthesize_from_labels` | ✅ |
| **FIX** (condition on real) | repaint path (`synth_p`<1, real masks) + `train.py` anatomy real-bg branch (excise) | ✅ |
| **FIT** (invert / twin) | — | ⛔ (`ncph`) |

**Control axis realized:** real ✅ (`store`) · parametric ✅ (SSM + `mri_physics` + strategies) · learned ⛔.

**Per-factor metrics:**

| metric | code | status |
|---|---|---|
| color coverage (W1, location/spread, by-vendor) | `core/data/analysis/synth_fidelity.py` | ✅ |
| shape coverage (embedding) | — | ⛔ (`uy4d`) |
| downstream Dice (cross-vendor) | `cardioseg/evaluation/validate` + `training/train.py` | ✅ |
| inverse recon error | — | ⛔ (`ncph`) |

**Reading it:** the whole **forward SAMPLE path is built** (parametric shape+color, all four bg strategies,
acquisition strategies, corruption chain), plus **FIX** (repaint + excised real-bg) and the **color-coverage
metric**. The gaps are the three that unlock the rest: **FIT** (`ncph`, the twin), **shape-coverage metric**
(`uy4d`), and the **learned shape prior** (`vpn5`) — with framing (`x8ne`) and per-vendor color (`ex1`) as
smaller fills.

## Current beads on this graph

`b6tb` (color/shape coverage of *generate*), `pwih` (augmentation MIX), `hpy` (richer shape+bg factor),
`ncph` (the FIT operator / twin), `uy4d` (shape-coverage metric). Each is a node or operator above,
not a free-floating pipeline.
