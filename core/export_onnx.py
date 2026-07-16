"""Export a trained 2D U-Net to ONNX (+ dynamic INT8 quantization) for deployment.

Lives with training because the ONNX is a model artifact — it's written next to model.pth
under the run dir. Gated by argmax parity vs PyTorch on a real patient; if a consumer
(e.g. cardioview's browser viewer) would segment differently, it isn't shipped.

    python -m core.export_onnx --run runs/acdc_aug
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from core.config import FLAGSHIP_REF
from core.data.static.mri.base import Dataset
from core.data.static.store.build import Build as store
from core.preprocessing.preprocess import SIZE, Preprocess
from core.registry import Registry
from core.run import Run

log = logging.getLogger("cardioseg.export_onnx")

PARITY_MIN = 99.0  # % argmax agreement required to ship the INT8 model (else keep FP32)
OPSET = 17         # ONNX opset for export


class ExportOnnx:
    """ONNX export for a trained 2D U-Net — the free helpers folded in as staticmethods: CPU weight load,
    PyTorch↔ONNX argmax parity, and the parity-gated export (+ INT8 quant)."""

    @staticmethod
    def _load_model(run: Path):
        """Load run weights on CPU for export — architecture from the run's saved config.json."""
        return Run.load_run(run, "cpu")[0]

    @staticmethod
    def _parity(model, onnx_path: Path, npz_path) -> float:
        """Per-slice argmax agreement (%) between PyTorch and an ONNX file on one consolidated subject."""
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        case = store.load_arrays(npz_path)
        imgs = np.stack([Preprocess.fit_square(s.astype(np.float32), SIZE, 0.0) for s in case["ed_img"]])
        agree = total = 0
        for s in imgs:
            x = s[None, None].astype(np.float32)
            with torch.no_grad():
                t = model(torch.from_numpy(x)).argmax(1)[0].numpy()
            o = sess.run(None, {"input": x})[0].argmax(1)[0]
            agree += int((t == o).sum())
            total += t.size
        return 100 * agree / total

    @staticmethod
    def export(run: Path, verify_dir: Path, *, quantize: bool = True,
               opset: int = OPSET, parity_min: float = PARITY_MIN) -> Path:
        """Write run/model.onnx from run/model.pth; INT8-quantize if it keeps parity. Returns the path."""
        run = Path(run)
        model = ExportOnnx._load_model(run)
        path = run / "model.onnx"
        torch.onnx.export(
            model, torch.randn(1, 1, SIZE, SIZE), str(path),
            input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=opset,
            dynamo=False,  # legacy exporter -> single self-contained .onnx (no .data sidecar)
        )
        p32 = ExportOnnx._parity(model, path, verify_dir)
        log.info(f"exported {path}  {path.stat().st_size / 1e6:.1f} MB  parity {p32:.3f}%")

        if quantize:
            q = run / "model.int8.onnx"
            quantize_dynamic(str(path), str(q), weight_type=QuantType.QInt8)
            pq = ExportOnnx._parity(model, q, verify_dir)
            log.info(f"quantized {q}  {q.stat().st_size / 1e6:.1f} MB  parity {pq:.3f}%")
            if pq >= parity_min:
                shutil.copyfile(q, path)
                log.info(f"-> model.onnx is INT8 (parity {pq:.2f}% >= {parity_min}%)")
            else:
                log.info(f"-> kept FP32 (int8 parity {pq:.2f}% < {parity_min}%)")
        return path

    @staticmethod
    def add_args(ap):
        ap.add_argument("--run", default=FLAGSHIP_REF, help="run dir holding model.pth")
        ap.add_argument("--verify", default=None, help="npz for the parity check (default: first ACDC subject)")
        ap.add_argument("--no-quantize", dest="quantize", action="store_false")
        ap.add_argument("--opset", type=int, default=OPSET, help=f"ONNX opset (default {OPSET})")
        ap.add_argument("--parity-min", type=float, default=PARITY_MIN,
                        help=f"%% argmax agreement to ship INT8 (default {PARITY_MIN})")

    @staticmethod
    def run(args):
        verify = args.verify if args.verify else store.load([Dataset.ACDC]).get_column("path")[0]
        ExportOnnx.export(Registry.resolve(args.run), verify, args.quantize,
                          opset=args.opset, parity_min=args.parity_min)
