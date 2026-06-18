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

### Quiz log
*(empty — append on demand)*

### Quiz log
*(empty — append on demand)*
