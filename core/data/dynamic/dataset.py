"""ACDC 2D-slice dataset for short-axis segmentation.

Cardiac MRI is highly anisotropic (10 mm slices vs ~1.5 mm in-plane), so a 2D
slice-wise model is the standard choice over naive 3D. Each item is one
short-axis slice + its 4-class mask (0 bg, 1 RV, 2 LV-myo, 3 LV-cavity). Frames
come from the preprocessed cache (in-plane resampled + z-scored). Slices are
centre padded/cropped to a fixed square so they batch.

Shapes through this file:
    cache volume      [D, H, W]   ->  per slice  [H, W]
    fit_square        [H, W]      ->  [size, size]
    __getitem__ item  img [1, size, size] float32,  mask [size, size] int64
    (a DataLoader then stacks B items -> img [B, 1, size, size], mask [B, size, size])

Split is PATIENT-LEVEL: every slice of a patient lands in the same fold, else
near-identical neighbouring slices leak across train/val and inflate Dice.
"""
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from core.data.static.store import load_arrays
from core.obs import Obs

# fit_square + SIZE are model-grid preprocessing primitives — they live in core now (shared by the
# training Dataset here and inference), single-sourced in core.preprocessing.preprocess.
from core.preprocessing.preprocess import SIZE, Preprocess
from core.types import Slice2D


class ACDCSliceDataset(Dataset):
    """All ED+ES short-axis slices from the given consolidated subjects, as (img, mask).

    Takes a list of npz paths from the data store (data/store.py) — each holds one subject's
    resampled + z-scored ed/es img+gt. Item: img float32 [1, size, size], mask int64 [size, size].
    Preloaded into RAM (slices are small; ~hundreds of subjects x 2 frames x ~10 slices x 256^2).
    """

    def __init__(
        self,
        npz_paths: list[str | Path],
        size: int = SIZE,
        frames: tuple[str, ...] = ("ED", "ES"),
        *,
        keep_empty: bool = False,
        augment: bool = False,
    ):
        self.size = size
        self.items: list[tuple[Slice2D, Slice2D]] = []   # (img[H,W] f32, mask[H,W] u8)
        self.owners: list[int] = []                      # per-slice index into npz_paths (for per-slice meta)
        self.frames = frames
        self.augment = augment
        for pi, p in enumerate(Obs.progress(npz_paths, f"load {'aug' if augment else 'val'} npz", total=len(npz_paths))):
            case = load_arrays(p)
            for tag in frames:
                img = case.get(f"{tag.lower()}_img")     # [D, H, W]
                gt = case.get(f"{tag.lower()}_gt")       # [D, H, W]
                if img is None:
                    continue
                for z in range(img.shape[0]):
                    m = gt[z]                             # [H, W]
                    if not keep_empty and m.max() == 0:
                        continue          # drop slices with no heart (apex/base air)
                    self.items.append((img[z].astype(np.float32), m.astype(np.uint8)))
                    self.owners.append(pi)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        img, m = self.items[i]
        img = Preprocess.fit_square(img, self.size, pad_value=0.0)          # [size, size]
        m = Preprocess.fit_square(m, self.size, pad_value=0)                # [size, size]
        # Augmentation is applied GPU-batched in the training loop (see training.augment), not
        # here — the per-item path stays cheap so DataLoader workers don't bottleneck the GPU.
        # img -> [1, size, size] (add channel); mask -> [size, size] int64
        return torch.from_numpy(img)[None], torch.from_numpy(m.astype(np.int64))

    @staticmethod
    def load_to_gpu(npz_paths, size: int = SIZE, device: str = "cuda", *, return_owners: bool = False):
        """Preload ALL slices into device memory as (imgs [N,1,size,size] f32, masks [N,size,size] uint8).

        The all-on-`device` dual of ACDCSliceDataset: slices are grid-fit ONCE here, then the training
        loop indexes these tensors on the GPU each epoch — zero per-epoch CPU / disk / host↔device copy
        (everything after setup runs on the GPU). The cardiac slice set fits VRAM easily (~3 GB at 256px
        on a 32 GB card). Masks kept uint8 to save VRAM; cast to long per batch. device='cpu' works too
        (CI / no-GPU) — same index-batched loop, just on CPU. `return_owners` also returns a per-slice
        LongTensor [N] indexing npz_paths — for attaching per-slice meta (e.g. the partial-label mask).

        (frames/keep_empty are ACDCSliceDataset knobs no caller ever overrode here — dropped rather than
        threaded as dead params; construct the dataset directly if you ever need them.)"""
        ds = ACDCSliceDataset(npz_paths, size=size)
        if not ds.items:
            z = torch.zeros((0, size, size))
            empty = (z[:, None].to(device), z.to(torch.uint8).to(device))
            return (*empty, torch.zeros(0, dtype=torch.long, device=device)) if return_owners else empty
        imgs = np.stack([Preprocess.fit_square(im, size, 0.0) for im, _ in ds.items]).astype(np.float32)
        msks = np.stack([Preprocess.fit_square(m, size, 0) for _, m in ds.items]).astype(np.uint8)
        X, Y = torch.from_numpy(imgs)[:, None].to(device), torch.from_numpy(msks).to(device)
        if return_owners:
            return X, Y, torch.tensor(ds.owners, dtype=torch.long, device=device)
        return X, Y

    @staticmethod
    def datasets(train_paths: list[str], val_paths: list[str], size: int = SIZE
                 ) -> tuple["ACDCSliceDataset", "ACDCSliceDataset"]:
        """(train_ds [augmented], val_ds) from two lists of consolidated-subject npz paths.

        The split itself is a query over the data store's meta frame (data/splits.py) — this just
        turns the chosen subject paths into augmented/plain slice datasets.
        """
        return (ACDCSliceDataset(train_paths, size=size, augment=True),
                ACDCSliceDataset(val_paths, size=size, augment=False))
