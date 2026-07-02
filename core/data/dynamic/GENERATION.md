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

Note the two objectives pull opposite ways, and that's fine: **forward/training wants wide coverage
(diversity > fidelity); inverse/twin wants tight fidelity.** Same process, opposite operations.

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

## Current beads on this graph

`b6tb` (color/shape coverage of *generate*), `pwih` (augmentation MIX), `hpy` (richer shape+bg factor),
`ncph` (the FIT operator / twin), `uy4d` (shape-coverage metric). Each is a node or operator above,
not a free-floating pipeline.
