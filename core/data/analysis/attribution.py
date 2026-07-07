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

import json
from pathlib import Path

import torch

from core.data.static.labels import CLASSES

_NAMES = ["bg"] + [nm for nm, _ in CLASSES.values()]      # ["bg","RV","LV-myo","LV-cav"]


def class_confusion(pred: torch.Tensor, gt: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Row-normalized confusion [n,n]: M[g,p] = fraction of GT-class-g voxels predicted as p. Rows for
    absent GT classes are left zero. Pure (no model) -> unit-testable."""
    M = torch.zeros(n_classes, n_classes)
    for g in range(n_classes):
        gm = gt == g
        tot = int(gm.sum())
        if tot == 0:
            continue
        pg = pred[gm]
        for p in range(n_classes):
            M[g, p] = (pg == p).float().mean()
    return M


def _predict(model, X: torch.Tensor, device: str, batch: int = 64) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return torch.cat([model(X[i:i + batch].to(device)).argmax(1).cpu()
                          for i in range(0, X.shape[0], batch)])


def _render(model, X, Y, pred, device, out_png: Path, n_classes: int, k: int = 4) -> bool:
    """real | GT | pred | saliency(cav) for k all-class slices. Saliency needs captum; returns whether
    it was drawn. Always writes the real/GT/pred panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    good = [i for i in range(Y.shape[0]) if set(Y[i].unique().tolist()) >= set(range(1, n_classes))][:k]
    if not good:
        good = list(range(min(k, Y.shape[0])))
    try:
        from captum.attr import Saliency
        def _fwd(x):
            return model(x).sum(dim=(2, 3))               # spatial-sum logits -> [B,C] for attribution
        sal = Saliency(_fwd)
    except Exception:
        sal = None

    rows = 4 if sal is not None else 3
    fig, ax = plt.subplots(rows, len(good), figsize=(3 * len(good), 3 * rows), squeeze=False)
    for c, i in enumerate(good):
        xi = X[i:i + 1].to(device)
        ax[0, c].imshow(X[i, 0].cpu(), cmap="gray"); ax[0, c].set_title("real"); ax[0, c].axis("off")
        ax[1, c].imshow(Y[i].cpu(), cmap="viridis", vmin=0, vmax=n_classes - 1); ax[1, c].set_title("GT"); ax[1, c].axis("off")
        ax[2, c].imshow(pred[i], cmap="viridis", vmin=0, vmax=n_classes - 1); ax[2, c].set_title("pred"); ax[2, c].axis("off")
        if sal is not None:
            a = sal.attribute(xi, target=n_classes - 1).abs()[0, 0].detach().cpu().numpy()
            ax[3, c].imshow(a, cmap="hot"); ax[3, c].set_title(f"saliency({_NAMES[-1]})"); ax[3, c].axis("off")
    fig.tight_layout(); fig.savefig(out_png, dpi=90); plt.close(fig)
    return sal is not None


def run_attribution(model, X: torch.Tensor, Y: torch.Tensor, n_classes: int, device: str,
                    out_dir: str | Path) -> dict:
    """Compute class confusion (always) + render attribution.png (saliency if captum). Writes
    attribution.json (confusion + per-class foreground-recall) to out_dir. Returns the summary dict."""
    out = Path(out_dir)
    pred = _predict(model, X, device)                    # on cpu
    conf = class_confusion(pred, Y.cpu(), n_classes)
    # foreground recall per class (diagonal) + the dominant leak (most-confused-with)
    summary = {
        "names": _NAMES[:n_classes],
        "confusion": [[round(float(conf[g, p]), 3) for p in range(n_classes)] for g in range(n_classes)],
        "recall": {_NAMES[g]: round(float(conf[g, g]), 3) for g in range(n_classes)},
    }
    has_sal = _render(model, X, Y, pred, device, out / "attribution.png", n_classes)
    summary["saliency"] = has_sal
    (out / "attribution.json").write_text(json.dumps(summary, indent=2))
    return summary


def _main():
    import argparse

    from core.config import FLAGSHIP_REF
    from core.data.dynamic.dataset import load_to_gpu
    from core.data.static import splits, store
    from core.hparams import TrainCfg
    from core.model import load_run
    from core.obs import setup
    from core.registry import resolve

    ap = argparse.ArgumentParser(description="Attribution diagnostic: confusion + saliency on a model.")
    ap.add_argument("--run", default=FLAGSHIP_REF, help="registry ref (alias|version|run-id) or run dir")
    ap.add_argument("--out", default=None, help="output dir (default: the resolved run dir)")
    a = ap.parse_args()
    setup()
    run_dir = resolve(a.run)
    model, cfg, device = load_run(run_dir)
    d = (cfg.generator.data if cfg else TrainCfg().generator.data)
    meta = store.load_cfg(d)                          # ALL preprocessing params (nyul/norm too)
    va = splits.model_val(d, meta)                   # coded split's val if set, else criteria
    X, Y = load_to_gpu(splits.paths(va), d.size, device)
    s = run_attribution(model, X, Y, (cfg.model.out_channels if cfg else 4), device, a.out or run_dir)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    _main()
