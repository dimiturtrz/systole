# Curriculum — foundations (maths)

The layer under everything. Working fluency (intuition + how to apply), **not
proofs**. Every topic is tied to *where it shows up in this pipeline* — that's the
point of learning it here rather than from a generic course.

Underpins both [../mri/](../mri/) (physics) and [../common/](../common/) (analysis).
Canonical text: **Deisenroth, Faisal & Ong, *Mathematics for Machine Learning***
(free PDF, [mml-book.github.io](https://mml-book.github.io/)) — structured exactly
as LA → calculus → probability, "for ML." Status: ✅ done · 🔄 doing · ⬜ planned.
Grounded in
[../../../research/deep_dives/2026-06-18_ml-geometry-application-curriculum.md](../../../research/deep_dives/2026-06-18_ml-geometry-application-curriculum.md).

---

## F1 · Linear algebra → [linear-algebra.md](F1_linear-algebra.md) ⬜
Vectors, matrices, matrix multiply, eigen/SVD, **affine transforms**.
Pipeline hooks: the NIfTI **affine** (voxel→mm), spacing/resampling, augmentation
rotations, **convolution as a linear operator**.

## F2 · Calculus & optimization → [calculus-optimization.md](F2_calculus-optimization.md) ⬜
Derivatives, gradients, chain rule (= **backprop**), gradient descent, convexity,
learning rate. Pipeline hooks: `loss.backward()` + Adam updating U-Net weights; why
the loss curve looks like it does; LR tuning.

## F3 · Probability & statistics → [probability-stats.md](F3_probability-stats.md) ⬜
Distributions, likelihood, **cross-entropy**, expectation/variance, estimation,
agreement statistics. Pipeline hooks: the CE half of Dice+CE; softmax as a
distribution; **Bland-Altman** for EF; cross-validation variance.

---

## Reference resources (benchmark)
- **Deisenroth et al. — *Mathematics for Machine Learning*** (free) — https://mml-book.github.io/ — the spine.
- **DeepLearning.AI — Math for ML & Data Science** — 3-course specialization (LA / calculus / prob-stats).
- **Math Academy — Mathematics for ML**; **dair-ai/Mathematics-for-ML** (resource list).
- 3Blue1Brown — *Essence of Linear Algebra* / *Essence of Calculus* (intuition, video).

*(We take the practitioner subset: enough to read a method/loss section and know
why, choose losses/metrics, and debug training/eval. Not measure theory, not proofs.)*
