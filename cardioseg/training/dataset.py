"""ACDC 2D-slice dataset for short-axis segmentation.

Cardiac MRI is highly anisotropic (10 mm slices vs ~1.5 mm in-plane), so a 2D
slice-wise model is the standard choice over naive 3D. Each item is one
short-axis slice + its 4-class mask (0 bg, 1 RV, 2 LV-myo, 3 LV-cavity). Frames
come from the preprocessed cache (in-plane resampled + z-scored). Slices are
centre padded/cropped to a fixed square so they batch.

Split is PATIENT-LEVEL: every slice of a patient lands in the same fold, else
near-identical neighbouring slices leak across train/val and inflate Dice.
"""
import numpy as np
from torch.utils.data import Dataset

from cardioseg.data.mri.data import acdc_cases
from cardioseg.preprocessing.preprocess import preprocess_case


def fit_square(arr, size, pad_value=0):
    """Centre pad/crop a [H,W] array to (size,size)."""
    h, w = arr.shape
    out = np.full((size, size), pad_value, dtype=arr.dtype)
    # source crop window (centred)
    sh, sw = max(0, (h - size) // 2), max(0, (w - size) // 2)
    src = arr[sh:sh + size, sw:sw + size]
    ch, cw = src.shape
    dh, dw = (size - ch) // 2, (size - cw) // 2
    out[dh:dh + ch, dw:dw + cw] = src
    return out


def split_patients(cases, val_frac=0.2, seed=0):
    """Deterministic patient-level train/val split."""
    cases = list(cases)
    idx = np.random.default_rng(seed).permutation(len(cases))
    n_val = max(1, int(round(len(cases) * val_frac)))
    val = {cases[i].name for i in idx[:n_val]}
    train = [c for c in cases if c.name not in val]
    val = [c for c in cases if c.name in val]
    return train, val


class ACDCSliceDataset(Dataset):
    """All ED+ES short-axis slices from the given patients, as (img, mask).

    img: float32 [1, size, size]; mask: int64 [size, size]. Preloaded into RAM
    (ACDC is small: ~100 patients x 2 frames x ~10 slices x 256^2 ~ 0.5 GB).
    """

    def __init__(self, patient_dirs, size=256, target_inplane=1.5,
                 frames=("ED", "ES"), keep_empty=False):
        self.size = size
        self.items = []          # list of (img[H,W] f32, mask[H,W] u8)
        self.frames = frames
        for pd in patient_dirs:
            c = preprocess_case(pd, target_inplane=target_inplane)
            for tag in frames:
                img = c.get(f"{tag.lower()}_img")
                gt = c.get(f"{tag.lower()}_gt")
                if img is None:
                    continue
                for z in range(img.shape[0]):
                    m = gt[z]
                    if not keep_empty and m.max() == 0:
                        continue          # drop slices with no heart (apex/base air)
                    self.items.append((img[z].astype(np.float32),
                                       m.astype(np.uint8)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        img, m = self.items[i]
        img = fit_square(img, self.size, pad_value=0.0)
        m = fit_square(m, self.size, pad_value=0)
        return torch.from_numpy(img)[None], torch.from_numpy(m.astype(np.int64))


def build_splits(size=256, val_frac=0.2, seed=0, n_patients=0):
    """Convenience: (train_ds, val_ds, train_dirs, val_dirs)."""
    cases = acdc_cases()
    if n_patients:
        cases = cases[:n_patients]
    train_dirs, val_dirs = split_patients(cases, val_frac, seed)
    return (ACDCSliceDataset(train_dirs, size=size),
            ACDCSliceDataset(val_dirs, size=size),
            train_dirs, val_dirs)
