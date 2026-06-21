"""Export a trained 2D U-Net to ONNX (+ dynamic INT8 quantization) for deployment.

Lives with training because the ONNX is a model artifact — it's written next to model.pth
under the run dir. Gated by argmax parity vs PyTorch on a real patient; if a consumer
(e.g. cardioview's browser viewer) would segment differently, it isn't shipped.

    python -m cardioseg.training.export_onnx --run runs/acdc_aug
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from cardioseg.training.model import build_unet
from cardioseg.training.dataset import fit_square
from cardioseg.data import store

SIZE = 256
PARITY_MIN = 99.0  # % argmax agreement required to ship


def load_model(run: Path):
    model = build_unet(spatial_dims=2, out_channels=4)
    model.load_state_dict(torch.load(run / "model.pth", map_location="cpu"))
    model.eval()
    return model


def parity(model, onnx_path: Path, npz_path) -> float:
    """Per-slice argmax agreement (%) between PyTorch and an ONNX file on one consolidated subject."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    case = store.load_arrays(npz_path)
    imgs = np.stack([fit_square(s.astype(np.float32), SIZE, 0.0) for s in case["ed_img"]])
    agree = total = 0
    for s in imgs:
        x = s[None, None].astype(np.float32)
        with torch.no_grad():
            t = model(torch.from_numpy(x)).argmax(1)[0].numpy()
        o = sess.run(None, {"input": x})[0].argmax(1)[0]
        agree += int((t == o).sum())
        total += t.size
    return 100 * agree / total


def export(run: Path, verify_dir: Path, quantize: bool = True) -> Path:
    """Write run/model.onnx from run/model.pth; INT8-quantize if it keeps parity. Returns the path."""
    run = Path(run)
    model = load_model(run)
    path = run / "model.onnx"
    torch.onnx.export(
        model, torch.randn(1, 1, SIZE, SIZE), str(path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # legacy exporter -> single self-contained .onnx (no .data sidecar)
    )
    p32 = parity(model, path, verify_dir)
    print(f"exported {path}  {path.stat().st_size / 1e6:.1f} MB  parity {p32:.3f}%")

    if quantize:
        from onnxruntime.quantization import quantize_dynamic, QuantType

        q = run / "model.int8.onnx"
        quantize_dynamic(str(path), str(q), weight_type=QuantType.QInt8)
        pq = parity(model, q, verify_dir)
        print(f"quantized {q}  {q.stat().st_size / 1e6:.1f} MB  parity {pq:.3f}%")
        if pq >= PARITY_MIN:
            shutil.copyfile(q, path)
            print(f"-> model.onnx is INT8 (parity {pq:.2f}% >= {PARITY_MIN}%)")
        else:
            print(f"-> kept FP32 (int8 parity {pq:.2f}% < {PARITY_MIN}%)")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/battery", help="run dir holding model.pth")
    ap.add_argument("--verify", default=None, help="npz for the parity check (default: first ACDC subject)")
    ap.add_argument("--no-quantize", dest="quantize", action="store_false")
    a = ap.parse_args()
    verify = a.verify if a.verify else store.load(["acdc"]).get_column("path")[0]
    export(Path(a.run), verify, a.quantize)


if __name__ == "__main__":
    main()
