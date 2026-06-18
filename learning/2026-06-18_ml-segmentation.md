# Phase D hands-on · ML / segmentation (Track A)

Connecting `materials/common/M_segmentation-theory.md` to the pipeline we actually
built (`cardioseg/training`, `cardioseg/evaluation`). Theory deepened in the
common doc; this file = the lessons + my understanding + quiz log (on demand).

Grounding numbers (our baseline, seed 0, no aug): Dice LV-cav 0.932 / myo 0.822 /
RV 0.859 (mean 0.871); EF MAE 3.1%. Code: `build_unet`, `train_acdc`, `validate`.

Lesson plan:
1. What the U-Net actually learns (segmentation as a learned function) ← here
2. U-Net architecture: encoder / decoder / skip connections — why
3. Training mechanics: the loss (Dice + CE), what a gradient step does
4. 2D vs 3D vs 2.5D — the anisotropy decision, concretely
5. Generalization: augmentation, overfitting, the domain-shift gap

---

## Lesson 1 — what the U-Net actually learns

**The task as a function.** Segmentation = a function `f` that takes one image
slice and returns, *for every voxel*, how likely it is to be each class:

```
f( image [1, H, W] )  ->  scores [4, H, W]      # 4 = bg, RV, myo, LV-cavity
```

`f` is the U-Net. It has ~millions of tunable numbers (**weights**). "Training"
= searching for weights so `f`'s output matches the ground-truth masks on the 80
training patients. Nothing about anatomy is hard-coded; the rules are *found*.

**Per voxel, but not in isolation.** Each output voxel is a classification (4
numbers -> softmax -> probabilities -> argmax -> one label). But the decision uses
a **neighbourhood**, not the single pixel: a bright voxel is "blood" only if a
muscle ring sits around it in the right place. The network sees that context
because convolutions look at local windows, and stacking them grows the
**receptive field** (how far out a voxel's decision can "see") — by the bottom of
the U-Net, one voxel's decision draws on a large patch. That's why a learned `f`
beats a brightness threshold: it judges *configuration*, not just intensity.

**What "the weights" are.** Convolution filters: shared little pattern-detectors
slid across the image. Early layers learn generic edges/textures; deeper layers
combine them into "blood-pool-ish blob", "myocardium boundary", "RV crescent".
Shared filters = the same detector works anywhere in the image (translation
equivariance) and needs far fewer parameters than a dense per-pixel net.

**In our code, concretely.**
- `build_unet(spatial_dims=2, out_channels=4)` — `out_channels=4` *is* the 4
  classes. Input `[B,1,H,W]` (1 = single MRI channel) -> output `[B,4,H,W]` scores.
- `train_acdc`: each step feeds a batch of slices, compares `f`'s scores to the
  true mask via the loss, nudges the weights. After 40 epochs the weights encode
  "what cardiac structures look like on ACDC".
- `validate`: `model(x).argmax(1)` turns scores -> a predicted label map; Dice
  measures how well that learned `f` reproduces held-out masks. LV-cav 0.932 =
  the cavity is learned well; myo 0.822 = the muscle boundary is harder (thin,
  fuzzy edge) — the model literally finds that class harder to pin down.

**One-line takeaway.** The U-Net is a big parameterised function from image to
per-voxel class scores; training picks its weights so it reproduces expert masks;
it works because convolutions let each voxel's label depend on learned *spatial
context*, not a fixed intensity rule.

## Lesson M (deltas) — what medical segmentation adds to general CV

For someone fluent in general deep learning, medical image segmentation is *mostly*
the same — here's the **delta**, the handful of things that are genuinely different
and that the interview/job cares about.

1. **Tiny labelled data.** ~100 patients, not a million images — expert annotation is
   slow + expensive. → U-Net's small-data design (skip connections, works on ~30
   images), heavy **augmentation**, sometimes transfer/self-supervision. Big models
   overfit instantly.
2. **3D + anisotropy.** Volumetric, and voxels aren't cubes (ACDC ~6–7×). → the **2D vs
   3D vs 2.5D** decision (2D wins on short-axis); you reason about it, you don't default.
3. **Severe class imbalance.** Background ≫ foreground in a slice. → **compound Dice+CE**:
   CE = MLE (smooth per-voxel gradient, see F3) but background-biased; **Dice** is
   overlap-normalised → robust to imbalance. Together = stability + balance.
4. **Patient-level splits — THE gotcha.** Slice-level splitting leaks near-identical
   neighbour slices across train/val → +5–10 fake Dice. Always split by patient.
5. **nnU-Net is the baseline-to-beat.** A *self-configuring* pipeline (reads the dataset
   fingerprint → sets patch size, norm, aug, depth, loss, postproc automatically). It
   matches/beats bespoke models on 23 challenges with **no architecture tuning**. The
   honest lesson: **a fancier net rarely wins; the value-add is the measurement +
   evaluation**, not the architecture.
6. **The label itself is fuzzy.** Single-expert GT with inter-rater variability (~±3% EF)
   → your "truth" has error bars (E2 floor). You can't meaningfully beat the labels.
7. **Segmentation isn't the deliverable — the clinical number is.** The mask feeds EF /
   thickness; so **boundary accuracy (mm) > raw overlap**, and the geometry/evaluation
   is where trust is won (the whole G + E tracks).

**Takeaway.** General CV gives you the U-Net and the training loop. Medical adds:
small fuzzy-labelled 3D-anisotropic class-imbalanced data, patient-level splits, a
self-configuring baseline (nnU-Net) that's hard to beat, and a pipeline whose real
output is a *trusted clinical measurement*, not a segmentation. The differentiator is
rigor downstream of the net, not the net.

### Quiz log
*(empty — append on demand)*
