"""Inference kernel — turn a trained model + a [D,H,W] volume into a label map. Shared by the
validation orchestration (cardioseg/evaluation/validate.py), the uncertainty/ensemble decomposition,
and the viewer (cardioview). Pure: model + array in, prediction out; no data-store or training deps.

`tta=True` averages the 4 in-plane flips; predict_volume_members keeps the per-flip stack as a cheap
K-member ensemble for the aleatoric/epistemic (BALD) split.
"""
import numpy as np
import torch

from core.preprocessing.preprocess import fit_square
from core.types import Volume

_FLIPS = ([], [2], [3], [2, 3])  # the 4 in-plane flips TTA averages over (identity, H, W, HW)


def _stack_slices(vol_img: Volume, size: int) -> np.ndarray:
    """Square-fit every slice of a [D, H, W] volume -> [D, size, size] float array (model input grid)."""
    return np.stack([fit_square(vol_img[z].astype(np.float32), size, 0.0) for z in range(vol_img.shape[0])])


def predict_volume_members(model, vol_img: Volume, size: int, device: str):
    """Run the 4 TTA flips and keep them as a cheap K-member ensemble for uncertainty decomposition.
    Returns (pred uint8 [D,size,size], mean_softmax [D,C,size,size], members [K,D,C,size,size]) on
    `device`. The members enable the aleatoric/epistemic (BALD) split; mean is their average."""
    model.eval()
    xs = _stack_slices(vol_img, size)
    with torch.no_grad():
        x = torch.from_numpy(xs)[:, None].to(device)          # [D, 1, size, size]
        d = x.shape[0]
        # ONE batched forward over all K flips ([K*D,1,H,W]) instead of K sequential calls — same math,
        # ~K× fewer kernel launches / python round-trips (the TTA test scores volume-by-volume × K).
        batched = torch.cat([torch.flip(x, dims) if dims else x for dims in _FLIPS], dim=0)
        probs = torch.softmax(model(batched), dim=1)          # [K*D, C, size, size]
        flips = [torch.flip(probs[i * d:(i + 1) * d], dims) if dims else probs[i * d:(i + 1) * d]
                 for i, dims in enumerate(_FLIPS)]             # un-flip each block back
        members = torch.stack(flips)                          # [K, D, C, size, size]
        mean = members.mean(0)                                # [D, C, size, size] mean softmax
        pred = mean.argmax(1).to(torch.uint8).cpu().numpy()
    return pred, mean, members


def predict_volume_probs(model, vol_img: Volume, size: int, device: str):
    """Mean-softmax over the 4 TTA flips — the shared inference primitive. Returns
    (pred uint8 [D,size,size], mean_softmax [D,C,size,size]). See predict_volume_members for the
    per-member stack used by the uncertainty decomposition."""
    pred, mean, _ = predict_volume_members(model, vol_img, size, device)
    return pred, mean


def predict_volume(model, vol_img: Volume, size: int, device: str, *, tta: bool = False) -> Volume:
    """Predict a label map [D, size, size] for one z-scored [D, H, W] volume.

    `tta=True` averages over the 4 in-plane flips (delegates to predict_volume_probs); `tta=False`
    is a single batched forward. argmax of the flip-sum == argmax of the mean, so results match."""
    if tta:
        pred, _ = predict_volume_probs(model, vol_img, size, device)
        return pred
    model.eval()
    xs = _stack_slices(vol_img, size)
    with torch.no_grad():
        x = torch.from_numpy(xs)[:, None].to(device)          # [D, 1, size, size]
        return model(x).argmax(1).cpu().numpy().astype(np.uint8)
