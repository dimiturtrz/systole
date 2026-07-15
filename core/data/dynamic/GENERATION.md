# Generation architecture — one process, many operations

> The synth pipelines share one skeleton. The forward path is a composable **Transform pipeline**
> (`pipeline.py`); generation is a **Source** behind the same seam as real data (`source.py` + the
> `ingest/` layer); the **FIT** operator lives in `inverse.py`; the **shape-coverage** metric in
> `analysis/shape_coverage.py`. Entry points are mapped at the bottom.

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
| **shape** (anatomy) | the label map: which voxel is RV-cav / myo / LV-cav / bg | `Anatomy.voxelize` (SSM meshes), or a real GT mask |
| **color** (appearance) | tissue T1/T2/PD **and** acquisition (field, TR, flip) → contrast | `mri_physics` + `Acquisition` strategy in `synth.py` (painter = `SynthPainter.synthesize_from_labels`) |
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
| **generate** (`b6tb` ✓, full-synth) | SAMPLE shape + color | SSM (Rodero) | randomized | full generation (shape+color) |
| **composite** (`uch6` ✓) | SAMPLE, UNION of sources | SSM ∪ pathology pool | randomized | many sources unioned into one dataset (`CompositeGenerator`) |
| **augmentation** (`pwih` ✓) | SAMPLE synth, MIX with real | mixed | randomized | synth as a robustness additive to real training |
| **inverse / digital twin** (`ncph`, `inverse.py`) | FIT | real (given) | **controlled** (fit) | recover interpretable qMRI params from a scan — loop built, but acquisition is **not identifiable** from one frame (see below) |
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

## Generation SOURCES — a composite all-synthetic dataset

The goal is **all-synthetic training data**, and no single generator reaches the whole real manifold.
So think of it as a **portfolio of sources**, each ENTERING the DAG at a different point with a different
**control degree**, all feeding the shared painter/corruption tail and **composed into one dataset**.
Each source covers a different region; the union covers more than any one.

| source | enters at | control | anatomy coverage | status |
|---|---|---|---|---|
| **fully parametric** | top (invent shape params → mesh → paint) | HIGH — every factor a knob | limited (needs a shape generator) | painter ✓, shape-invent ✗ |
| **SSM (Rodero)** | shape = pre-built mesh → voxelize → paint | MED — SSM mode weights | healthy + MILD pathology (±3SD) | ✓ (pool_1000) |
| **pathology pool** | SSM meshes pushed into the DCM/HCM/RV tail | MED — pathology knobs | dilated/thick tail SSM ±3SD misses | ✓ (`pool_pathology`, unioned via `CompositeGenerator`, `uch6`) |
| **label-space edit** | shape = deform existing label maps | MED — deform params | variations; pathology via dilation | build (`vpn5`) |
| **MRXCAT** | whole thing (torso+heart+physics) → we consume its **label `.vti`**, paint with our engine | LOW–MED — pathology knobs | whole-FOV + pathology + structured bg | `hpy` — adapter `core/data/dynamic/mrxcat.py` ✓ (`to_canonical`/`load_vti_labels`/`build_pool`; remap geometrically verified — myo ring encloses LV-cav). Tool = external checkout (public ETH repo, MIT-cited): `git clone https://gitlab.ethz.ch/ibt-cmr-public/mrxcat-2.0.git external/mrxcat2 && git -C external/mrxcat2 checkout 9f396a9` — kept in gitignored `external/`, never vendored (fetch to be folded into the mrxcat generation CLI, bd cardiac-seg-8pfl); XCAT torso Duke-gated but example bundled. NB MRXCAT paints myo UNIFORM by construction (`fixLVTexture` meanLV) — confirms our myo over-spread is low-res-PV, not physics |
| **learned prior** | shape = sample a learned model | LOW — latent | the real manifold incl pathology | future (`vpn5` option) |

**Building the pools (committed CLI, bd 8pfl)** — the offline builds are reproducible subcommands, not ad-hoc REPL:

```bash
# SSM (Rodero) anatomy — healthy pool, then the DCM/HCM/RV pathology pool
# (offline builders dispatch through `python -m core.data <group> <subcmd>`; bd ox6p)
python -m core.data build-pool convert-binary --mesh-dir <data>/volumetric/meshes/raw
python -m core.data build-pool build-pool --mesh-dir <data>/volumetric/meshes/raw --out <data>/volumetric/meshes/processed/rodero_anatomy/pool.npz
python -m core.data build-pool build-pathology-pool --pool <data>/volumetric/meshes/processed/rodero_anatomy/pool.npz --out <data>/volumetric/meshes/processed/rodero_anatomy/pool_pathology.npz
# MRXCAT — fetch the external tool (pinned), then heart-only / whole-FOV / SSM×MRXCAT pools
python -m core.data mrxcat fetch
python -m core.data mrxcat build-pool --vti-dir external/mrxcat2/<vti> --out <data>/mrxcat/processed/pool.npz
python -m core.data mrxcat build-fov-pool --vti-dir external/mrxcat2/<vti> --out <data>/mrxcat/processed/fov_pool.npz
python -m core.data mrxcat build-ssm-fov-pool --rodero-pool <data>/volumetric/meshes/processed/rodero_anatomy/pool.npz --vti-dir external/mrxcat2/<vti> --out <data>/mrxcat/processed/ssm_fov_pool.npz
```

**Composition is cheap** (union of label pools → the painter is shared): `CompositeGenerator` concatenates
source pools; `SynthComposite` (`ingest/splits/synth_composite.py`) is the split that trains on it. The
VALUE is *diverse sources*, not one bigger source. Coverage is measured per source and on the union
(`shape_coverage`, `static_compare`), so we can see which source fills which gap — e.g. SSM covers normals
(NOR 0.49), the DCM/RV tail needs the pathology pool / label-space / learned / MRXCAT. Control degree is a
*feature*: high-control sources (parametric) for targeted gaps, whole-thing sources (MRXCAT) for breadth.

**Result — coverage ≠ Dice:** the SSM ∪ pathology composite pushes shape coverage ~0.78 → ~0.94 (the
pathology pool fills the DCM/HCM tail SSM misses) but zero-real TEST Dice does **not** move (within 2-seed
noise). Closing coverage does not close Dice, so the remaining ceiling is shape/color **fidelity**
(boundary/texture detail), not coverage — the next fidelity lever is within-slice texture or a learned
shape prior (`vpn5`), not more coverage sources. Unioning the full pool thrashes VRAM, so `DynamicSource`
caps per-source residency.

## Implementation map (what's built, where)

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
| **SAMPLE** (generate) | `generator.py`: `Generator` / `CompositeGenerator`; forward as a composable Transform list in `pipeline.py`; `SynthPainter.synthesize_from_labels` | ✅ |
| **FIX** (condition on real) | repaint path (`synth_p`<1, real masks) + `train.py` anatomy real-bg branch (excise) | ✅ |
| **FIT** (invert / twin) | `inverse.py`: `Inverse` — differentiable render, gradient descent to θ | 🟡 loop built + converges, but the one-frame heart fit is **degenerate** (`ncph`/`ixea`, see note) |

**Control axis realized:** real ✅ (`store`) · parametric ✅ (SSM + `mri_physics` + strategies) · learned ⛔.

**Per-factor metrics:**

| metric | code | status |
|---|---|---|
| color coverage (W1, location/spread, by-vendor) | `core/data/analysis/synth_fidelity.py` | ✅ |
| shape coverage (embedding) | `core/data/analysis/shape_coverage.py` (`python -m core.data.analysis shape-coverage`) | ✅ (`uy4d`) |
| downstream Dice (cross-vendor) | `cardioseg/evaluation/validate` + `training/train.py` | ✅ |
| inverse recon error | `inverse.py` (the FIT residual = the probe) | 🟡 (`ncph`/`ixea`) |

**Reading it:** the whole **forward SAMPLE path is built** (parametric shape+color, all four bg strategies,
acquisition strategies, corruption chain, composite union), plus **FIX** (repaint + excised real-bg), both
coverage metrics (**color** + **shape**), and the **FIT** loop (`inverse.py`). What's left is not
mechanism but *fidelity + identifiability*: the composite showed coverage is saturated yet Dice-flat, so
the open levers are a **learned shape prior** (`vpn5`) and texture fidelity for the forward path, and
**multi-acquisition input** (`5ev5`) to make the twin identifiable — with framing (`x8ne`) and per-vendor
color (`ex1`) as smaller fills.

## Identifiability of the FIT operator (`ncph`/`ixea`)

The FIT loop is mechanically correct (differentiable render, gradient descent, converges) but the
one-frame heart fit is **degenerate by construction**. The heart has only TWO tissue levels (blood,
myocardium; RV-cav == LV-cav == blood). Uncalibrated MRI intensity is only comparable after a gain/bias
normalization, and an affine map takes two levels onto two levels **exactly** for *any* acquisition — so
the acquisition signal (the blood/myo contrast ratio) is normalized away and acquisition is **not
identifiable at all** from one frame (not even a single param). Breaking the degeneracy needs one of:
(a) **multiple acquisitions** of the same anatomy (varied flip/TR — real qMRI, `5ev5`); (b) **absolute-
calibrated intensity** (which uncalibrated cardiac MRI doesn't give — the whole domain problem); or
(c) **≥3 known tissue levels**. So the digital twin needs multi-acquisition input; the probe use (does
the forward physics span this scan?) still works from the recon residual.

## Entry points (the committed CLIs)

Generation and its analysis are reproducible commands (no REPL). The three group dispatchers:

```bash
# data / pool builders  — python -m core.data <cmd>
python -m core.data build-pool …      # SSM (Rodero) anatomy pools (Anatomy)
python -m core.data mrxcat …          # MRXCAT fetch + label pools (Mrxcat)
python -m core.data twin …            # the FIT operator / inverse fit (Inverse)
python -m core.data consolidate …     # real-store build; also: reference, lock-testsets, kaggle-ef

# generation analysis   — python -m core.data.analysis <cmd>
python -m core.data.analysis shape-coverage …   # shape embedding coverage (uy4d)
python -m core.data.analysis synth-fidelity …   # per-class colour W1
python -m core.data.analysis static-compare …   # source-vs-real / union coverage
#   also: attribution, eda, render, sim2real
```

**Splits** — a generation source is consumed as a named, hash-frozen split family (`core.data.ingest`),
the same seam real data flows through: `static_main`, `static_all`, `synth_main`, `synth_composite`
(registered in `ingest/splits/__init__.py`; `Splits.load_split(name)` → resolve via `SplitResolver`).
Train picks one with `--split <name>`.

## Current beads on this graph

Closed: `b6tb` (generate coverage), `pwih` (augmentation MIX), `uy4d` (shape-coverage metric), `uch6`
(composite sources), `8pfl` (pool-build CLI), `mirs` (real-bg excise bug). Open: `hpy` (MRXCAT richer
shape+bg), `ncph`/`5ev5` (the twin / multi-acquisition identifiability), `vpn5` (learned/label-space
pathology shape prior), `x8ne` (framing/FOV margin), `ex1` (per-vendor conditioned color). Each is a node
or operator above, not a free-floating pipeline.
