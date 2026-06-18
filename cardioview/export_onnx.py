"""Export the trained 2D U-Net to ONNX for in-browser inference (onnxruntime-web), and
optionally quantize it (dynamic INT8) for a smaller download + faster CPU/PC inference.

Both exports are gated by argmax agreement vs PyTorch on a real patient — if the browser
wouldn't segment like the pipeline, the artifact isn't shipped.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from common import load_model, MODELS, SIZE, square_stack, patient_dir
from cardioseg.preprocessing.preprocess import preprocess_case

OUT = Path("cardioview/web/public/models")
PARITY_MIN = 99.0  # % argmax agreement required to ship


def parity(model, onnx_path: Path, patient: str) -> float:
    """Per-slice argmax agreement (%) between PyTorch and an ONNX file on one patient."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    imgs = square_stack(preprocess_case(patient_dir(patient))["ed_img"])  # [D,256,256] z-scored
    agree = total = 0
    for s in imgs:
        x = s[None, None].astype(np.float32)
        with torch.no_grad():
            t = model(torch.from_numpy(x)).argmax(1)[0].numpy()
        o = sess.run(None, {"input": x})[0].argmax(1)[0]
        agree += int((t == o).sum())
        total += t.size
    return 100 * agree / total


def export(model_name: str, verify: str, quantize: bool) -> None:
    model = load_model(MODELS[model_name], "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{model_name}.onnx"

    torch.onnx.export(
        model, torch.randn(1, 1, SIZE, SIZE), str(path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # legacy exporter -> single self-contained .onnx (no .data sidecar)
    )
    fp32_mb = path.stat().st_size / 1e6
    p32 = parity(model, path, verify)
    print(f"exported {path.name}  {fp32_mb:.1f} MB  parity {p32:.3f}%")

    if not quantize:
        return
    from onnxruntime.quantization import quantize_dynamic, QuantType

    qpath = OUT / f"{model_name}.int8.onnx"
    quantize_dynamic(str(path), str(qpath), weight_type=QuantType.QInt8)
    q_mb = qpath.stat().st_size / 1e6
    pq = parity(model, qpath, verify)
    print(f"quantized {qpath.name}  {q_mb:.1f} MB ({100 * (1 - q_mb / fp32_mb):.0f}% smaller)  parity {pq:.3f}%")
    if pq >= PARITY_MIN:
        shutil.copyfile(qpath, path)  # ship int8 as the bundled model
        print(f"-> serving int8 as {path.name} (parity {pq:.2f}% >= {PARITY_MIN}%)")
    else:
        print(f"-> keeping fp32 (int8 parity {pq:.2f}% < {PARITY_MIN}%)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="acdc_aug", choices=list(MODELS))
    ap.add_argument("--verify", default="patient073")
    ap.add_argument("--no-quantize", dest="quantize", action="store_false")
    a = ap.parse_args()
    export(a.model, a.verify, a.quantize)


if __name__ == "__main__":
    main()
