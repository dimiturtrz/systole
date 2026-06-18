# Phase D hands-on · geometry / volumetry (Track B)

Connecting `materials/common/G_geometry-and-volumetry.md` to the pipeline
(`cardioseg/evaluation/measure.py`, `cardioseg/analysis/viz.py`). Theory deepened
in the common doc; this file = lessons + my understanding + quiz log (on demand).

Started after Track A (the ML makes the mask; geometry turns it into the number).

Lesson plan:
1. voxel grid -> physical volume (spacing, anisotropy, Simpson's = stack of disks)
2. marching cubes: voxel mask -> triangle surface mesh
3. mesh operations: wall thickness (surface-to-surface), surface area
4. where geometry decides the number: partial volume, basal/apical conventions
5. Hausdorff distance as geometry (boundary distance, not overlap)

---

## Lesson G1 — voxel grid → volume → EF (the analytical core)

**A voxel is a little box.** The NIfTI affine / spacing says its physical size:
`dx × dy × dz` mm (ACDC ~1.56 × 1.56 × 10). So one voxel's volume is the product:

```
voxel_volume = dx · dy · dz   (mm³)        # measure.voxel_volume_ml = prod(spacing)/1000  -> mL
```

**A region's volume = count its voxels × box volume.**
```
V(label) = N_voxels(label) · dx · dy · dz   # measure.label_volume_ml
```
This is a **Riemann sum**: the true (continuous) volume `∭ 1[inside] dV` approximated
by summing identical grid boxes. The mask is the indicator function sampled on the grid.

**Simpson's method (the clinical name) is the same thing.** Clinicians compute LV
volume as a **stack of disks**: for each short-axis slice, area × slice thickness,
summed:
```
V = Σ_slices  A_i · dz       where  A_i = (pixels in slice i) · dx · dy
```
Substitute and it collapses to `N_voxels · dx·dy·dz` — **voxel counting *is* Simpson's
summation-of-disks.** Same integral, two vocabularies (ML "count voxels" = clinical
"sum the disks"). Knowing they're identical is the bridge between the two worlds.

**EF is a ratio, so spacing cancels.**
```
EF = (EDV − ESV) / EDV
```
Multiply EDV and ESV by the same `dx·dy·dz` and it divides out. Consequence: EF is
**scale-invariant** — a wrong-but-constant spacing still gives the right EF (why our
earlier hardcoded-spacing EF was fine), but any **mL** number needs the real spacing.

**Where the analytical error lives** (not in the arithmetic — in the discretization):
- **Partial volume** — a voxel straddling blood/muscle gets one label; worst where the
  surface is oblique to the grid (apex taper, base). The Riemann sum's step error.
- **Through-plane coarseness** — dz=10mm means big boxes through-plane; the dominant
  volume error (the integrand changes a lot within one box).
- **Boundaries of integration** — where to stop basally (valve plane) and apically
  (cap beyond last slice) = choosing the integration limits; convention shifts mL.

**Takeaway.** Volume is an integral; the mask is the integrand sampled on a grid;
voxel-counting = Simpson's disks = the Riemann sum. EF divides two such sums so the
voxel size cancels. The error isn't arithmetic — it's how coarsely the grid samples
a curved surface (partial volume), worst through-plane and at apex/base.

## Lesson G2 — marching cubes (mask → triangle surface)

**The problem.** A mask is a blocky binary grid (inside=1 / outside=0). We want the
**boundary as a smooth surface** — a triangle mesh — not a Minecraft blob. (The
"hard part = boundary" from before, now an explicit geometric object.)

**Isosurface idea.** Treat the mask as a scalar field; the surface is where it
crosses an **isovalue** — for a binary mask, **0.5** (halfway between out=0/in=1).
`skimage...marching_cubes(lv, level=0.5, spacing=sp)`.

**The algorithm** (Lorensen & Cline 1987):
1. **March a cube** through the grid — 8 voxel corners.
2. Each corner **in or out** (above/below 0.5) → 8 bits → **2^8 = 256** configs.
3. A precomputed **lookup table** maps each config → which cube *edges* the surface
   crosses + how to wire them into triangles.
4. **Interpolate** each vertex along its edge to where field = isovalue (binary =
   edge midpoint). Result: **vertices [N,3]** + **faces [M,3]** (triangle indices).

**Spacing makes it physical** — `spacing=(z,y,x)` puts vertices in **mm**, inheriting
anisotropy.

**Real run — patient001 LV cavity (ED):**
```
12104 voxels  ->  4842 vertices, 9586 triangles
surface area = 248.8 cm^2        <- a number voxel-counting CANNOT give
bbox (z,y,x) = 85 x 77 x 78 mm   <- 85 mm of z across only 10 slices (10 mm steps)
```
The z-line is the sting: through-plane the mesh is **stepped** (10 mm levels) while
in-plane it's smooth (1.56 mm) — anisotropy you can now *see* in the geometry.
Smoothing (G3) cleans the steps; it can't add missing data.

**Unlocks (real geometry now, not counting):** surface area (above); wall thickness
= endo→epi surface distance (G3); Hausdorff = boundary distance (G4); rendering.

**Ambiguity.** Some configs are **saddle** (ambiguous) → holes if wired
inconsistently; variants (asymptotic decider, marching tetrahedra) fix it; skimage
handles it.

**Takeaway.** Marching cubes turns a label volume into the **boundary surface** by
classifying each 8-corner cube and table-looking-up the crossing pattern; spacing
puts it in mm; anisotropy shows as through-plane stepping. Every later geometry
(area, thickness, Hausdorff, render) runs on this object.

## Lesson G3 — three layers (don't conflate them)

G3 first lumped "mesh processing" with "wall thickness" — wrong, they're unrelated.
Split into three clearly-labelled layers:

### Layer 1 — VISUALIZATION (graphics; zero cardiac content)
Making the mesh look good / render fast. A game engine does the identical thing to
any model. **Not knowledge — polish.** Lives in **cardioview**.
- **Smoothing (Taubin).** Raw marching-cubes mesh is blocky/stepped → nudge vertices
  toward neighbours. (Plain **Laplacian** shrinks the model with repetition; **Taubin**
  alternates shrink/inflate to smooth without shrink — a graphics gotcha, still not
  cardiac.) cardioview: `smooth_taubin(n_iter=24, pass_band=0.05)`.
- **Decimation.** Drop triangles (quadric error metric) → smaller/faster. `decimate(0.7)`.
- It **slightly changes the geometry** → never measure on it.

### Layer 2 — CARDIAC MEASUREMENT (the actual domain knowledge — memorise this)
What a cardiologist reads off the anatomy.
- **Wall thickness** = how thick the heart muscle is = distance from its **inner**
  surface (endo, blood side) to its **outer** surface (epi). Normal LV ~6–12 mm.
- **Systolic thickening** — the muscle *thickens* as it squeezes (ED→ES). Our run:
  max 12.9 → 15.6 mm. A dead/scarred segment does **not** thicken — that's how you spot it.
- **Thresholds:** HCM (hypertrophic) wall **≥15 mm**; **thinning** = old infarct/scar;
  **AHA 17-segment** = thickness per region.
- **Why it beats EF:** EF is one whole-heart scalar; thickness is a **regional map** —
  catches one bad wall segment a normal global EF hides. Needs the **myo** label EF ignores.

### Layer 3 — the math underneath (CS tools; serve Layer 2)
Neutral algorithms that *compute* the above.
- **Marching cubes** (G2) — mask → surface.
- **Distance transform** ("distance to nearest other-label voxel") — turns "endo and
  epi surfaces" into a thickness number. Real run used it: half-width × 2 inside the myo ring.

### Real run — patient001 (DCM), mid short-axis slice
```
ED: wall ~ median 6.2 mm, max 12.9 mm    (normal LV ~6-12 mm at ED)
ES: wall ~ median 6.2 mm, max 15.6 mm    (thickens in systole)
```

**The split that matters**

| thing | what it is | where | knowledge? |
|---|---|---|---|
| smooth, decimate | make it pretty | cardioview | no — graphics |
| wall thickness, thickening, HCM | what the heart is doing | clinical / cardioseg | **yes — cardiology** |
| marching cubes, distance transform | compute the above | shared CS tools | math, not domain |

**Takeaway.** Layer 1 changes the geometry → so measure Layer 2 (thickness, EF) on the
**raw voxels**, apply Layer 1 **only to the picture**. Same heart, two outputs: honest
numbers from voxels, smooth render from the mesh. The cardiac lesson here is **wall
thickness** (what it is, systolic thickening, regional map) — the rest is polish + tools.

## Lesson G4 — Hausdorff (boundary distance as a metric)

**Why Dice isn't enough.** Dice measures **overlap/area**. It's blind to *where* the
boundary is. Real demo (patient001 LV cavity, GT vs perturbed "predictions"):
```
perfect  (gt vs gt)        Dice 1.000   HD   0.0 mm
boundary shifted ~5 mm     Dice 0.870   HD   6.6 mm     <- both catch it
gt + 1 stray FP blob       Dice 1.000   HD 208.5 mm     <- Dice BLIND, HD screams
```
The stray-blob row is the lesson: a false-positive speck 200 mm away **doesn't change
overlap** → Dice stays 1.000 → totally misses it. You need a **boundary** metric.

**Hausdorff distance.** Boundary-to-boundary distance: for every point on
prediction's surface, distance to the nearest point on GT's surface; take the **max**;
symmetrise (max of both directions). "Worst-case how far is my boundary from truth,"
in **mm**. (This is Q2's surface-to-surface distance, formalised.)

**Computed via the distance transform** (the Q2 tool): `dt = distance_transform_edt(~G)`
gives distance-to-G's-surface everywhere; sample it on P's surface; take the max.
`cardioseg/evaluation/evaluate.py:hausdorff` does exactly this, spacing-aware (mm).

**The catch — outlier sensitivity (the Q1 max-vs-median lesson again).** HD is the
**single worst point**, so ONE stray voxel → HD=208 mm. Catastrophically fragile. Fix:
**HD95** = 95th percentile of the surface distances (drop the top 5%) → robust to
specks, still catches real boundary error. Good cardiac HD95 ~2–5 mm.

**Dice vs HD — report both, they're orthogonal:**
- **Dice** = how much overlap (bulk/volume-ish). Misses non-overlapping FPs + smooth offsets.
- **Hausdorff/HD95** = how far the boundary is off (shape/surface, worst-case).
A model can ace one and fail the other; the two rows above show each failure mode.

**Connects to the postprocessing follow-up (cardiac-seg-w4l).** `keep_largest` deletes
exactly the stray FP blob that detonates HD (208 mm → gone). So cleaning predictions
helps **HD a lot**, EF a little — another reason that follow-up matters.

**Takeaway.** Dice = overlap, blind to boundary location and to non-overlapping
errors. Hausdorff = worst boundary distance (mm) — catches what Dice can't, but one
outlier wrecks it, so use **HD95**. Same median-vs-max robustness lesson as wall
thickness. Boundary is the hard part (G2); these are how you *score* the boundary.

## Lesson G5 — registration (aligning two images/shapes)

**The problem.** Two views of the same heart sit in different positions/shapes: ED vs
ES (same heart, *contracted* → different shape), slices drifting between breath-holds,
or MRI vs CT (different modality). **Registration = find the transform that aligns one
("moving") onto the other ("fixed").**

**Transform families (increasing flexibility):**
- **Rigid** (6 DOF: 3 rotate + 3 translate) — move/rotate, no shape change. Motion
  correction; same rigid object.
- **Affine** (12 DOF: + scale + shear) — global linear stretch. (Same affine math as F1.)
- **Deformable / non-rigid** — a **displacement vector per voxel** (a warp field). The
  heart *changes shape* ED→ES (contracts, twists), so ED↔ES alignment needs this.

**How (it's an optimisation — the F2 track).** Search the transform that **maximises
image similarity** (or minimises a distance) between fixed & moving, + a **regulariser**
keeping the warp smooth/plausible. Similarity metrics: SSD/cross-correlation
(same modality), **mutual information** (cross-modality, where intensities differ).

**Why it matters for cardiac (this is the bridge past EF):**
- The ED→ES **deformation field = how each piece of muscle moved = strain / regional
  motion.** That's the rung-1 modelling beyond EF — the field *is* the regional function
  (a dead segment doesn't move → shows up in the field).
- **Label propagation:** register ED→ES, warp the ED mask → an ES mask. Semi-automatic
  segmentation / consistency across frames.
- **Motion correction** (breath-hold slice drift); **cross-modality fusion** (MRI↔CT);
  **atlas / shape models** (G6, cross-subject).

**Tools:** **SimpleITK** (the standard), ANTs, elastix.

**Where we are.** Our EF pipeline does **no** registration — EF only needs ED & ES
volumes independently. G5 is the **bridge** to strain/motion (regional function) and to
cross-modality — awareness + how it works, not in the current build.

**Takeaway.** Registration = optimise a transform (rigid → affine → deformable) to align
two images by similarity. The deformable **warp field is the payload**: it turns two
static frames into *motion* (strain) — the step from "measure EF" toward "model the
heart's function." Same optimisation (F2) + affine (F1) foundations, applied to alignment.

### Quiz log
- [G1–G2 · 2026-06-18](quizzes/common/G1-G2_2026-06-18.md) — ~24.5/30 ≈ 82% (strong).
  Big gap: *why a ratio is robust* (correlated-bias cancellation, not just "small error").
- [G1–G6 full track · 2026-06-18](quizzes/common/G1-G6_2026-06-18.md) — ~41/60 ≈ 68% (solid pass).
  Gaps: spacing-cancels-as-ratio (Q1); verts ∝ surface (Q4); HD95 removes outliers (Q8);
  rigid reports zero strain (Q10); **MI = statistical dependence** (Q11, re-read G5).
