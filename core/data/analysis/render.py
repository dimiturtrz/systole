"""Visual synth-vs-real diagnostic — render real slices next to synth (same masks) + per-class
intensity stats. The 'look before you train' check that repeatedly caught what a metric hid (washed
blobs, cartoon backgrounds, blood-too-bright). Saves a PNG grid. Companion to synth_fidelity (numbers)
and sim2real (per-vendor fit)."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib as mpl
import numpy as np
import torch

mpl.use("Agg")
import matplotlib.pyplot as plt

from core.data.dynamic.dataset import ACDCSliceDataset
from core.data.dynamic.synth import FlatBgCfg, PartitionBgCfg, SynthCfg, SynthPainter
from core.data.static import splits
from core.data.static.labels import CLASSES
from core.data.static.store.build import Build as store
from core.hparams import TrainCfg

log = logging.getLogger("cardioseg.render")

_NAMES = ["bg"] + [nm for nm, _ in CLASSES.values()]


class Render:
    """Visual synth-vs-real diagnostic — the free renderer folded in as a staticmethod."""

    @staticmethod
    def render_synth_vs_real(out_png: str | Path = ".staging/synth_diag.png", k: int = 4, seed: int = 0):
        """Load k real val slices (all heart classes present), generate synth from their masks, save a grid
        (real | mask | synth-flat | synth-partition) + print per-class real-vs-synth intensity stats."""
        torch.manual_seed(seed); np.random.seed(seed)
        data_cfg = TrainCfg().generator.data
        n = len(CLASSES) + 1
        meta = store.load_cfg(data_cfg, workers=4)              # ALL preprocessing params (nyul/norm too)
        val_split = splits.ModelSplit(data_cfg, meta).val                   # held-out real slices (coded split's val if set)
        X, Y = ACDCSliceDataset.load_to_gpu(splits.Splits.paths(val_split), data_cfg.size, "cpu")
        all_class_slices = [i for i in range(Y.shape[0]) if set(Y[i].unique().tolist()) >= set(range(1, n))][:k]
        X, Y = X[all_class_slices], Y[all_class_slices]
        torch.manual_seed(1); synth_flat, _ = SynthPainter.synthesize_from_labels(Y, SynthCfg(synth_p=1.0, bg=FlatBgCfg()), n)
        torch.manual_seed(2); synth_partition, _ = SynthPainter.synthesize_from_labels(Y, SynthCfg(synth_p=1.0, bg=PartitionBgCfg()), n, real_img=X)

        rows = [("real", X[:, 0]), ("mask", Y.float()), ("synth flat", synth_flat[:, 0]), ("synth partition", synth_partition[:, 0])]
        fig, ax = plt.subplots(len(rows), len(all_class_slices), figsize=(3 * len(all_class_slices), 3 * len(rows)), squeeze=False)
        for r, (name, vol) in enumerate(rows):
            for c in range(len(all_class_slices)):
                ax[r, c].imshow(vol[c].cpu().numpy(), cmap="viridis" if "mask" in name else "gray")
                ax[r, c].axis("off")
                if c == 0:
                    ax[r, c].set_ylabel(name)
        out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out_png, dpi=90); plt.close(fig)
        log.info(f"wrote {out_png}\nPER-CLASS mean±std (z), real vs synth-partition:")
        for class_index in range(n):
            real_intensities, synth_intensities = X[:, 0][class_index == Y], synth_partition[:, 0][class_index == Y]
            log.info(f"  {_NAMES[class_index]:8} real {real_intensities.mean():+.2f}±{real_intensities.std():.2f}   synth {synth_intensities.mean():+.2f}±{synth_intensities.std():.2f}")


    @staticmethod
    def add_args(ap):
        pass

    @staticmethod
    def run(args):  # pragma: no cover
        Render.render_synth_vs_real()
