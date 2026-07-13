"""Mask post-processing for predictions.

A 2D slice model leaves stray false-positive islands (a few voxels of "RV" in the lung,
a speck of "LV-cav" off the blood pool). They barely move Dice but inflate volumes — and
EF is a *ratio* of volumes, so they bias it. Keeping only the largest connected component
of each class is the standard cheap clean-up (and what nnU-Net tests for, too).

CPU/scipy only: a once-per-volume clean-up (~15 ms on a [16,256,256] mask) is negligible beside
GPU inference, and for a ~1 MB mask a cupy/cucim GPU port is host<->device-transfer-bound — no
speedup for a second code path. (The `gpu` extra exists for `losses.py`'s boundary-loss distance
transform, not for this.)
"""
import numpy as np
from scipy.ndimage import label as _cc_label

from core.data.static.labels import FOREGROUND
from core.types import Mask


class Postprocess:
    """Largest-connected-component clean-up (the free helpers folded in as staticmethods)."""

    @staticmethod
    def largest_cc_binary(binary: np.ndarray) -> np.ndarray:
        """Largest connected component of a boolean volume (drop stray islands). CPU/scipy.
        Shared by mesh export and cardioview geometry; the per-class path below inlines it."""
        lab, n = _cc_label(binary)
        if n <= 1:
            return binary
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0  # ignore background
        return lab == int(sizes.argmax())

    @staticmethod
    def largest_cc_per_class(mask: Mask, labels: tuple[int, ...] = FOREGROUND) -> Mask:
        """Keep only the largest 3D connected component of each foreground class.

        mask: [D, H, W] integer label map (classes disjoint, as from argmax). Returns a cleaned
        copy — false-positive islands dropped, the main structure kept."""
        out = np.zeros_like(mask)
        for lab in labels:
            binary = mask == lab
            if not binary.any():
                continue
            out[Postprocess.largest_cc_binary(binary)] = lab
        return out
