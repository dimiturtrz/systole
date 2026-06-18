# cardioview

3D visualization of the [`cardioseg`](../cardioseg) model's results — an inference/demo view
of the pipeline. **The product is the browser viewer in [`web/`](web/)** (TS + vtk.js):
rotatable colored chambers (LV cavity / myocardium / RV), the **beating cardiac cycle**, and
**EDV / ESV / LVEF** read out against ground truth. Plus **import your own**: upload a
`.nii.gz` and it's segmented in-browser (ONNX) and rendered.

![cardioview demo](docs/media/demo.gif)

## Setup

Nothing data-derived is committed (ACDC licensing) — the hearts and the model are built from
the pipeline. Two ways in:

### A — canned demo hearts (full pipeline)
```bash
# 1. deps (from repo root) — installs cardioseg + cardioview Python deps
pip install -e .

# 2. data: register for ACDC, then point paths.yaml at it (data stays outside the repo)
cp paths.example.yaml paths.yaml      # edit data.raw -> .../acdc  (dir holding training/)

# 3. train the segmentation model (see cardioseg/README) -> runs/acdc/model.pth
python -m cardioseg.training.train --acdc --epochs 40

# 4. bake the web assets (use the model you trained: --model acdc or acdc_aug — the viewer
#    follows it via the manifest). Hearts come from paths.yaml (cardioview.hearts).
python cardioview/export_onnx.py --model acdc         # -> web/public/models/acdc.onnx
python cardioview/export_web.py --mode animate --model acdc   # -> web/public/data/*.glb + manifest.json

# 5. run the viewer
cd cardioview/web && npm install && npm run dev        # http://localhost:5173
```

### B — just explore, no data/training
```bash
cd cardioview/web && npm install && npm run dev
```
No canned hearts or bundled model, but the panel's **import .onnx** + **import scan (.nii.gz)**
let you drop in your own model and scans (segmented in-browser). See [web/README](web/README.md).

> Requirements per project: **mri-sim** is the simplest (just `npm install && npm run dev`,
> no data/model); **cardioview** needs the model + ACDC for the canned hearts (above);
> **cardioseg** is the pipeline (`pip install -e .` + ACDC).

## Python tools (offline)
- `export_web.py` — segment ED/ES (or every 4D frame, `--mode animate`) → chamber `.glb` + EF manifest
- `export_onnx.py` — thin: runs `cardioseg.training.export_onnx` (the canonical ONNX export +
  INT8 quant + parity gate, written next to `model.pth`) and copies the artifact to the web
- `render_overlay.py` / `render_volume.py` — desktop pyvista quick-looks (screenshot or `--interactive`)

**Honesty:** predictions are tagged `held-out` vs `TRAIN-seen` (deterministic split check), EF
is shown vs ground truth, and model false-positive specks aren't cleaned up. Volumes come from
full-res masks × real spacing — the render grid never touches the numbers.

Tracked in beads (`bd show cardiac-seg-5nh`).
