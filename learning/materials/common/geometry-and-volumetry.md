# B3 · Geometry & volumetry — labels → a clinical number

The connective tissue of the whole project: how segmentation masks become millilitres
and then EF. Same geometry across modalities.

## Voxel count → volume
```
Volume (mL) = N_voxels × dx × dy × dz / 1000
```
- `dx, dy` = in-plane spacing (mm), `dz` = slice thickness (mm); ÷1000 converts mm³→mL.
- ACDC typical voxel: 1.5 × 1.5 × 8 mm = **18 mm³ = 0.018 mL**.
- **Spacing comes from the NIfTI header** — never assume isotropic. (Our `measure.py`
  already does spacing-aware volume + EF.)

## Simpson's method (summation of disks) — same thing
The standard CMR volumetry method:
```
V = Σᵢ Aᵢ × dz        where  Aᵢ = (pixels of the structure in slice i) × dx × dy
```
Sum cross-sectional areas across the short-axis slices. **Voxel counting *is* Simpson's
method** when each slice contributes `area = N_pixels × dx × dy`. Do it on the ED frame
and the ES frame → EDV, ESV → EF.

Conventions that bite:
- **Basal slice inclusion** — a common rule: include a basal slice only if myocardium
  surrounds ≥50% of the cavity. Affects volume; match the dataset/eval convention.
- **Apical cap** — the apex beyond the most apical slice is approximated (cone) or
  ignored → small systematic underestimate (~5–10 mL).
- **Papillary muscles** — included in the LV cavity by convention (~12% of EDV); be
  consistent ED↔ES. (See [cardiac-anatomy-and-cycle.md](cardiac-anatomy-and-cycle.md).)

## Surface meshing (marching cubes)
Converts a voxel mask → a triangulated surface. **Not needed for EF** (Simpson's
suffices), but used for: 3D visualization, surface-based metrics (Hausdorff distance —
see [evaluation-theory.md](evaluation-theory.md)), and wall-thickness. (Our `viz.py`
does marching-cubes → STL.)

## Wall thickness
Shortest distance from endocardial to epicardial surface, per point. Needs both LV
cavity (3) and LV myocardium (2) labels. Normal LV wall ~6–12 mm at ED; HCM ≥15 mm.
Systolic thickening = (ES − ED thickness)/ED; low thickening → ischemia/scar.

## Why short-axis suits volumetry
Slices are perpendicular to the LV long axis, so each is a true cross-section of the
cavity. Summing disk areas is geometrically exact (within slice-thickness error) for
any ventricle shape — dilated, hypertrophied, infarcted. That's why CMR is the
volumetric gold standard.
