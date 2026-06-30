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
uv sync --all-extras

# 2. data: register for M&M-2 (training, https://www.ub.edu/mnms-2/) + ACDC (the canned hearts,
#    https://www.creatis.insa-lyon.fr/Challenge/acdc/); point paths.yaml at them
cp paths.example.yaml paths.yaml      # edit data.raw -> .../acdc; M&M-2 sits beside it

# 3. train the flagship (see cardioseg/README) -> runs/gen/model.pth
#    = core.config.FLAGSHIP_RUN: pooled M&M-2 + M&Ms-1, ACDC + Canon held out.

# 4. bake the web assets (the viewer follows the model via the manifest). Hearts come from
#    paths.yaml (cardioview.hearts) — ACDC patients the flagship never trained on.
#    --model defaults to 'gen' (cardioview/common.py DEFAULT_MODEL).
python cardioview/export_onnx.py --model gen          # -> web/public/models/gen.onnx (flagship; default)
python cardioview/export_web.py --mode animate --model gen    # -> web/public/data/*.glb + manifest.json

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
> no data/model); **cardioview** needs a trained model + ACDC patients for the canned hearts
> (above); **cardioseg** is the pipeline (`uv sync --all-extras` + M&M-2/ACDC).

## Python tools (offline)
- `export_web.py` — segment ED/ES (or every 4D frame, `--mode animate`) → chamber `.glb` + EF manifest
- `export_onnx.py` — thin: runs `cardioseg.training.export_onnx` (the canonical ONNX export +
  INT8 quant + parity gate, written next to `model.pth`) and copies the artifact to the web
- `render_overlay.py` / `render_volume.py` — desktop pyvista quick-looks (screenshot or `--interactive`)

**Honesty:** predictions are tagged `held-out` vs `TRAIN-seen` (deterministic split check), EF
is shown vs ground truth, and model false-positive specks aren't cleaned up. Volumes come from
full-res masks × real spacing — the render grid never touches the numbers.

Tracked in beads (`bd show cardiac-seg-5nh`).
