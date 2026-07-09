"""Visual synth-vs-real diagnostic — render real slices next to synth (same masks) + per-class
intensity stats. The 'look before you train' check that repeatedly caught what a metric hid (washed
blobs, cartoon backgrounds, blood-too-bright). Saves a PNG grid. Companion to synth_fidelity (numbers)
and sim2real (per-vendor fit)."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.data.dynamic.dataset import ACDCSliceDataset
from core.data.dynamic.synth import FlatBgCfg, PartitionBgCfg, SynthCfg, SynthPainter
from core.data.static import splits
from core.data.static.labels import CLASSES
from core.data.static.store.build import Build as store
from core.hparams import TrainCfg
from core.obs import Obs

log = logging.getLogger("cardioseg.render")

_NAMES = ["bg"] + [nm for nm, _ in CLASSES.values()]


class Render:
    """Visual synth-vs-real diagnostic — the free renderer folded in as a staticmethod."""

    @staticmethod
    def render_synth_vs_real(out_png: str | Path = ".staging/synth_diag.png", k: int = 4, seed: int = 0):
        """Load k real val slices (all heart classes present), generate synth from their masks, save a grid
        (real | mask | synth-flat | synth-partition) + print per-class real-vs-synth intensity stats."""
        torch.manual_seed(seed); np.random.seed(seed)
        d = TrainCfg().generator.data
        n = len(CLASSES) + 1
        meta = store.load_cfg(d, workers=4)              # ALL preprocessing params (nyul/norm too)
        va = splits.Splits.model_val(d, meta)                   # held-out real slices (coded split's val if set)
        X, Y = ACDCSliceDataset.load_to_gpu(splits.Splits.paths(va), d.size, "cpu")
        good = [i for i in range(Y.shape[0]) if set(Y[i].unique().tolist()) >= set(range(1, n))][:k]
        X, Y = X[good], Y[good]
        torch.manual_seed(1); Sf, _ = SynthPainter.synthesize_from_labels(Y, SynthCfg(synth_p=1.0, bg=FlatBgCfg()), n)
        torch.manual_seed(2); Sp, _ = SynthPainter.synthesize_from_labels(Y, SynthCfg(synth_p=1.0, bg=PartitionBgCfg()), n, real_img=X)

        rows = [("real", X[:, 0]), ("mask", Y.float()), ("synth flat", Sf[:, 0]), ("synth partition", Sp[:, 0])]
        fig, ax = plt.subplots(len(rows), len(good), figsize=(3 * len(good), 3 * len(rows)), squeeze=False)
        for r, (name, vol) in enumerate(rows):
            for c in range(len(good)):
                ax[r, c].imshow(vol[c].cpu().numpy(), cmap="viridis" if "mask" in name else "gray")
                ax[r, c].axis("off")
                if c == 0:
                    ax[r, c].set_ylabel(name)
        out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out_png, dpi=90); plt.close(fig)
        log.info(f"wrote {out_png}\nPER-CLASS mean±std (z), real vs synth-partition:")
        for c in range(n):
            rm, sm = X[:, 0][Y == c], Sp[:, 0][Y == c]
            log.info(f"  {_NAMES[c]:8} real {rm.mean():+.2f}±{rm.std():.2f}   synth {sm.mean():+.2f}±{sm.std():.2f}")


def main():
    argparse.ArgumentParser(description="render synth-vs-real diagnostic panels").parse_args()
    Obs.setup()
    Render.render_synth_vs_real()


if __name__ == "__main__":
    main()
