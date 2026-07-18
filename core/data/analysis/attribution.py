"""Model attribution / diagnostic — what did the model learn, and where does it systematically fail?

Two views on a trained model over an eval set:
  1. CLASS CONFUSION (always): per GT class, the fraction predicted as each label. Surfaces directional
     mistakes (e.g. the pure-synth model predicts background on 51% of RV — systematic under-seg), which
     a single mean-Dice number hides.
  2. SALIENCY (optional, needs captum): which input pixels drive a class prediction — reveals shortcuts
     (e.g. keying only on the bright LV-cavity blob, ignoring the thin RV).

Reusable on any model (registry ref | run dir) and optionally run at the end of training (before ONNX),
so every run ships an attribution.png + confusion next to its card. captum is an optional dep
(`pip install .[diag]`); without it, confusion still runs, saliency is skipped.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib as mpl
import torch
from captum.attr import Saliency
from jaxtyping import Float, Integer
from monai.networks.nets import UNet

mpl.use("Agg")
import matplotlib.pyplot as plt

from core.config import FLAGSHIP_REF
from core.data.dynamic.dataset import ACDCSliceDataset
from core.data.static import splits
from core.data.static.labels import CLASSES
from core.data.static.store.build import Build as store
from core.hparams import TrainCfg
from core.registry import Registry
from core.run import Run

log = logging.getLogger("cardioseg.attribution")

_NAMES = ["bg"] + [nm for nm, _ in CLASSES.values()]      # ["bg","RV","LV-myo","LV-cav"]


class Attribution:
    """Model-interpretability diagnostic (confusion recall + captum saliency) for a trained model.
    Holds model + device + n_classes as STATE (bd 01fh: they were threaded as args); .attribute(X, Y,
    out_dir) takes only the data + output location that vary. The model-free helpers (class_confusion, _predict)
    are folded in as staticmethods."""

    def __init__(self, model: UNet, device: str, n_classes: int) -> None:
        self.model, self.device, self.n_classes = model, device, n_classes

    @staticmethod
    def class_confusion(pred: Integer[torch.Tensor, "*grid"], gt: Integer[torch.Tensor, "*grid"],
                        n_classes: int) -> Float[torch.Tensor, "*n *n"]:
        """Row-normalized confusion [n,n]: M[g,p] = fraction of GT-class-g voxels predicted as p. Rows for
        absent GT classes are left zero. Pure (no model) -> unit-testable."""
        confusion = torch.zeros(n_classes, n_classes)
        for gt_class in range(n_classes):
            gt_mask = gt == gt_class
            total = int(gt_mask.sum())
            if total == 0:
                continue
            pred_at_gt = pred[gt_mask]
            for pred_class in range(n_classes):
                confusion[gt_class, pred_class] = (pred_at_gt == pred_class).float().mean()
        return confusion

    @staticmethod
    def _predict(model: UNet, X: Float[torch.Tensor, "*batch *c *h *w"], device: str,
                 batch: int = 64) -> Integer[torch.Tensor, "*batch *h *w"]:
        model.eval()
        with torch.no_grad():
            return torch.cat([model(X[i:i + batch].to(device)).argmax(1).cpu()
                              for i in range(0, X.shape[0], batch)])

    def attribute(self, X: Float[torch.Tensor, "*batch *c *h *w"], Y: Integer[torch.Tensor, "*batch *h *w"],
                  out_dir: str | Path) -> dict[str, Any]:
        """Compute class confusion (always) + render attribution.png (saliency if captum). Writes
        attribution.json (confusion + per-class foreground-recall) to out_dir. Returns the summary dict."""
        out_dir_path = Path(out_dir)
        pred = self._predict(self.model, X, self.device)          # on cpu
        confusion = self.class_confusion(pred, Y.cpu(), self.n_classes)
        # foreground recall per class (diagonal) + the dominant leak (most-confused-with)
        summary = {
            "names": _NAMES[:self.n_classes],
            "confusion": [[round(float(confusion[gt_class, pred_class]), 3) for pred_class in range(self.n_classes)]
                          for gt_class in range(self.n_classes)],
            "recall": {_NAMES[gt_class]: round(float(confusion[gt_class, gt_class]), 3)
                       for gt_class in range(self.n_classes)},
        }
        has_saliency = self._render(X, Y, pred, out_dir_path / "attribution.png")
        summary["saliency"] = has_saliency
        (out_dir_path / "attribution.json").write_text(json.dumps(summary, indent=2))
        return summary

    def _render(self, X: Float[torch.Tensor, "*batch *c *h *w"], Y: Integer[torch.Tensor, "*batch *h *w"],
                 pred: Integer[torch.Tensor, "*batch *h *w"], out_png: Path, k: int = 4) -> bool:
        """real | GT | pred | saliency(cav) for k all-class slices. Saliency needs captum; returns whether
        it was drawn. Always writes the real/GT/pred panel."""
        all_class_slices = [i for i in range(Y.shape[0])
                            if set(Y[i].unique().tolist()) >= set(range(1, self.n_classes))][:k]
        if not all_class_slices:
            all_class_slices = list(range(min(k, Y.shape[0])))
        def _fwd(x: torch.Tensor) -> torch.Tensor:
            return self.model(x).sum(dim=(2, 3))             # spatial-sum logits -> [B,C] for attribution
        saliency = Saliency(_fwd)

        rows = 4 if saliency is not None else 3
        vmax = self.n_classes - 1
        fig, ax = plt.subplots(rows, len(all_class_slices), figsize=(3 * len(all_class_slices), 3 * rows),
                               squeeze=False)
        def _panel(row: int, c: int, img: Any, title: str, **kw: Any) -> None:
            ax[row, c].imshow(img, **kw); ax[row, c].set_title(title); ax[row, c].axis("off")
        for c, i in enumerate(all_class_slices):
            slice_input = X[i:i + 1].to(self.device)
            _panel(0, c, X[i, 0].cpu(), "real", cmap="gray")
            _panel(1, c, Y[i].cpu(), "GT", cmap="viridis", vmin=0, vmax=vmax)
            _panel(2, c, pred[i], "pred", cmap="viridis", vmin=0, vmax=vmax)
            if saliency is not None:
                saliency_map = (saliency.attribute(slice_input, target=self.n_classes - 1)
                                .abs()[0, 0].detach().cpu().numpy())
                _panel(3, c, saliency_map, f"saliency({_NAMES[-1]})", cmap="hot")
        fig.tight_layout(); fig.savefig(out_png, dpi=90); plt.close(fig)
        return saliency is not None

    @staticmethod
    def add_args(ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--run", default=FLAGSHIP_REF, help="registry ref (alias|version|run-id) or run dir")
        ap.add_argument("--out", default=None, help="output dir (default: the resolved run dir)")

    @staticmethod
    def run(args: argparse.Namespace) -> None:  # pragma: no cover
        run_dir = Registry.resolve(args.run)
        model, cfg, device = Run.load_run(run_dir)
        data_cfg = (cfg.generator.data if cfg else TrainCfg().generator.data)
        meta = store.load_cfg(data_cfg)                          # ALL preprocessing params (nyul/norm too)
        val_split = splits.ModelSplit(data_cfg, meta).val                   # coded split's val if set, else criteria
        X, Y = ACDCSliceDataset.load_to_gpu(splits.Splits.paths(val_split), data_cfg.size, device)
        summary = Attribution(model, device, cfg.model.out_channels if cfg else 4).attribute(X, Y, args.out or run_dir)
        log.info(json.dumps(summary, indent=2))
