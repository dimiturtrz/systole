"""Mask post-processing for predictions.

A 2D slice model leaves stray false-positive islands (a few voxels of "RV" in the lung,
a speck of "LV-cav" off the blood pool). They barely move Dice but inflate volumes — and
EF is a *ratio* of volumes, so they bias it. Keeping only the largest connected component
of each class is the standard cheap clean-up (and what nnU-Net tests for, too).
"""
import numpy as np
from scipy.ndimage import label as _cc_label

from core.data.static.labels import FOREGROUND
from core.types import Mask


class Postprocess:
    """Largest-connected-component clean-up (the free helpers folded in as staticmethods): the shared
    binary largest-CC, the cucim GPU capability gate, and the per-class CPU/GPU dispatch."""

    @staticmethod
    def largest_cc_binary(binary: np.ndarray) -> np.ndarray:
        """Largest connected component of a boolean volume (drop stray islands). CPU/scipy.
        Shared by mesh export and cardioview geometry; the per-class GPU/CPU paths below inline
        the same logic on their respective array libs."""
        lab, n = _cc_label(binary)
        if n <= 1:
            return binary
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0  # ignore background
        return lab == int(sizes.argmax())

    @staticmethod
    def gpu_cc():
        """cucim GPU connected-components (the linux GPU lane) if importable, else None -> scipy CPU.
        Detected once at import — a single capability gate, no per-call branching beyond the dispatch."""
        try:
            import cupy  # noqa: F401, PLC0415  # pragma: no cover
            from cucim.skimage.measure import label  # noqa: PLC0415  # pragma: no cover
            return label  # pragma: no cover
        except ImportError:                        # cupy/cucim not installed (windows / no-GPU) -> CPU fallback
            return None

    @staticmethod
    def _largest_cc_gpu(mask: Mask, labels) -> Mask:  # pragma: no cover  (linux GPU lane only — no cucim on CI)
        """GPU largest-CC via cupy + cucim (linux lane). Same result as the scipy path; back to numpy."""
        import cupy as cp  # noqa: PLC0415
        m = cp.asarray(mask)
        out = cp.zeros_like(m)
        for lab in labels:
            binary = m == lab
            if not bool(binary.any()):
                continue
            cc = _CUCIM_LABEL(binary)
            sizes = cp.bincount(cc.ravel())
            sizes[0] = 0
            out[cc == int(sizes.argmax())] = lab
        return cp.asnumpy(out)

    @staticmethod
    def largest_cc_per_class(mask: Mask, labels: tuple[int, ...] = FOREGROUND) -> Mask:
        """Keep only the largest 3D connected component of each foreground class.

        mask: [D, H, W] integer label map (classes disjoint, as from argmax). Returns a cleaned
        copy — false-positive islands dropped, the main structure kept. Uses the cucim GPU path when
        available (linux), else scipy CPU — identical output either way."""
        if _CUCIM_LABEL is not None:               # pragma: no cover  (GPU dispatch — CPU path tested)
            return Postprocess._largest_cc_gpu(mask, labels)
        out = np.zeros_like(mask)
        for lab in labels:
            binary = mask == lab
            if not binary.any():
                continue
            out[Postprocess.largest_cc_binary(binary)] = lab
        return out


_CUCIM_LABEL = Postprocess.gpu_cc()
