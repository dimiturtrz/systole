"""Mask post-processing for predictions.

A 2D slice model leaves stray false-positive islands (a few voxels of "RV" in the lung,
a speck of "LV-cav" off the blood pool). They barely move Dice but inflate volumes — and
EF is a *ratio* of volumes, so they bias it. Keeping only the largest connected component
of each class is the standard cheap clean-up (and what nnU-Net tests for, too).
"""
import numpy as np
from scipy.ndimage import label as _cc_label

from cardioseg.types import Mask
from cardioseg.labels import FOREGROUND


def largest_cc_per_class(mask: Mask, labels: tuple[int, ...] = FOREGROUND) -> Mask:
    """Keep only the largest 3D connected component of each foreground class.

    mask: [D, H, W] integer label map (classes disjoint, as from argmax). Returns a cleaned
    copy — false-positive islands dropped, the main structure kept.
    """
    out = np.zeros_like(mask)
    for lab in labels:
        binary = mask == lab
        if not binary.any():
            continue
        cc, n = _cc_label(binary)
        if n <= 1:
            out[binary] = lab
            continue
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0  # ignore background
        out[cc == int(sizes.argmax())] = lab
    return out
