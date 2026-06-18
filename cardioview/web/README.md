# cardioview-web

Browser viewer for the segmented hearts — TypeScript + Vite + [vtk.js](https://kitware.github.io/vtk-js/),
same stack as `mri-sim`. Pure rendering: all inference/geometry is precomputed offline by
[`../export_web.py`](../export_web.py) and shipped as **glb meshes + an EF manifest**.

**Status: foundation.** Rotatable 3D colored chambers (LV cavity / myocardium / RV) loaded
from glb; side panel with **EDV (full) · ESV (empty) · LVEF %** + category, stroke volume,
GT EF and the pred-vs-GT error, plus a `held-out` / `train-seen` honesty tag. Patient picker
and ED/ES toggle. Myocardium is re-made translucent in-viewer (glTF drops opacity) with depth
peeling so the cavities read through it.

## Run
```bash
# 1) precompute assets (Python env with cardioseg + pyvista), from the repo root:
PYTHONPATH=. python cardioview/export_web.py --patients patient006 patient009 patient010
# 2) the web app:
cd cardioview/web && npm install && npm run dev      # http://localhost:5173
npm test      # vitest (metrics + manifest self-consistency)
npm run smoke # puppeteer: asserts chambers actually paint
```
Assets (`public/data/*.glb`, `manifest.json`) are gitignored — regenerate with step 1.

## Architecture
- `src/metrics.ts` — pure measurement helpers (EF identity, formatting) — unit-tested
- `src/manifest.ts` — manifest types + fetch
- `src/viewer.ts` — vtk.js scene + GLTFImporter loader (trackball, depth peeling)
- `src/panel.ts` — controls + EDV/ESV/EF readout (view glue)

## Next
Beating-cycle animation (precomputed 4D phases) · patient gallery · in-browser ONNX
segmentation of an uploaded `.nii.gz`. Tracked in beads (`bd show cardiac-seg-5nh`).
