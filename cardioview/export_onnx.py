"""Place the trained ONNX into the web viewer.

The actual export (+ INT8 quant + parity gate) lives with training:
cardioseg.training.export_onnx writes runs/<name>/model.onnx. Here we just run it and
copy the artifact to web/public/models/<name>.onnx (served as the bundled model).

    python cardioview/export_onnx.py --model acdc_aug
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from cardioseg.training.export_onnx import export as build_onnx
from cardioseg.data import store
from common import MODELS, DEFAULT_MODEL

OUT = Path("cardioview/web/public/models")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    ap.add_argument("--verify", default=None)
    ap.add_argument("--no-quantize", dest="quantize", action="store_false")
    a = ap.parse_args()

    run = Path(MODELS[a.model]).parent  # runs/<name>/model.pth -> runs/<name>
    # parity check needs a CONSOLIDATED-STORE npz (not a raw patient dir) — same source as
    # cardioseg.training.export_onnx's own default.
    verify = a.verify if a.verify else store.load(["acdc"]).get_column("path")[0]
    onnx = build_onnx(run, verify, a.quantize)  # runs/<name>/model.onnx

    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{a.model}.onnx"
    shutil.copyfile(onnx, dst)
    print(f"-> bundled web model: {dst}")


if __name__ == "__main__":
    main()
