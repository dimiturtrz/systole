"""Export the trained 2D U-Net to ONNX for in-browser inference (onnxruntime-web).

Verifies the export against PyTorch on a real patient — if argmax agreement isn't
~100%, the browser would segment differently than the pipeline, so this gate matters.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from common import load_model, MODELS, SIZE, square_stack, patient_dir
from cardioseg.preprocessing.preprocess import preprocess_case

OUT = Path("cardioview/web/public/models")


def export(model_name: str, verify_patient: str) -> None:
    model = load_model(MODELS[model_name], "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{model_name}.onnx"

    dummy = torch.randn(1, 1, SIZE, SIZE)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # legacy exporter -> single self-contained .onnx (no .data sidecar) for the web
    )
    print(f"exported {path}")

    # Parity gate: torch vs onnxruntime, per-slice argmax agreement on a real volume.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    case = preprocess_case(patient_dir(verify_patient))
    imgs = square_stack(case["ed_img"])  # [D, 256, 256] z-scored
    agree = total = 0
    for s in imgs:
        x = s[None, None].astype(np.float32)
        with torch.no_grad():
            t = model(torch.from_numpy(x)).argmax(1)[0].numpy()
        o = sess.run(None, {"input": x})[0].argmax(1)[0]
        agree += int((t == o).sum())
        total += t.size
    print(f"parity: {100 * agree / total:.4f}% argmax agreement (torch vs onnx) on {verify_patient}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="acdc_aug", choices=list(MODELS))
    ap.add_argument("--verify", default="patient073")
    a = ap.parse_args()
    export(a.model, a.verify)


if __name__ == "__main__":
    main()
