"""Inference kernel — turn a trained model + a [D,H,W] volume into a label map. Shared by the
validation orchestration (cardioseg/evaluation/validate.py), the uncertainty/ensemble decomposition,
and the viewer (cardioview). Pure: model + array in, prediction out; no data-store or training deps.

`tta=True` averages the 4 in-plane flips; predict_volume_members keeps the per-flip stack as a cheap
K-member ensemble for the aleatoric/epistemic (BALD) split.
"""
import numpy as np
import torch
from jaxtyping import Float, UInt8

from core.preprocessing.preprocess import Preprocess
from core.types import Volume, shapecheck

_FLIPS = ([], [2], [3], [2, 3])  # the 4 in-plane flips TTA averages over (identity, H, W, HW)


class Inference:
    """A trained model bound to its input grid + device: construct once, predict many [D,H,W] volumes.
    `model`, `size`, and `device` are the fixed inference session; each call supplies only the volume.

    `logit_bias` (opt-in, off by default) adds a per-class constant to the logits pre-softmax — a
    log-prior on the class mix. A positive RV bias recovers the RV-omission tail (raw RV softmax present
    but out-competed by background at argmax on small apical slices; bd nttu.5/nttu.7) without retraining.
    Val-fit, applied post-hoc — a labelled prior like the EF calibration (tb58), NOT a physics claim."""

    def __init__(self, model, size: int, device: str, logit_bias=None):
        self.model, self.size, self.device = model, size, device
        self.logit_bias = (
            torch.as_tensor(logit_bias, dtype=torch.float32, device=device).reshape(1, -1, 1, 1)
            if logit_bias is not None else None)

    def _bias(self, logits: Float[torch.Tensor, "n c h w"]) -> Float[torch.Tensor, "n c h w"]:
        """Add the per-class logit prior (a no-op when unset). Broadcast [1,C,1,1] over [N,C,H,W]."""
        return logits if self.logit_bias is None else logits + self.logit_bias

    def _stack_slices(self, vol_img: Volume) -> np.ndarray:
        """Square-fit every slice of a [D, H, W] volume -> [D, size, size] float array (model input grid)."""
        return np.stack([Preprocess.fit_square(vol_img[z].astype(np.float32), self.size, 0.0)
                         for z in range(vol_img.shape[0])])

    def predict_volume_members(self, vol_img: Float[np.ndarray, "d h w"]):
        """Run the 4 TTA flips and keep them as a cheap K-member ensemble for uncertainty decomposition.
        Returns (pred uint8 [D,size,size], mean_softmax [D,C,size,size], members [K,D,C,size,size]) on
        `device`. The members enable the aleatoric/epistemic (BALD) split; mean is their average."""
        self.model.eval()
        xs = self._stack_slices(vol_img)
        with torch.no_grad():
            x = torch.from_numpy(xs)[:, None].to(self.device)     # [D, 1, size, size]
            d = x.shape[0]
            # ONE batched forward over all K flips ([K*D,1,H,W]) instead of K sequential calls — same math,
            # ~K× fewer kernel launches / python round-trips (the TTA test scores volume-by-volume × K).
            batched = torch.cat([torch.flip(x, dims) if dims else x for dims in _FLIPS], dim=0)
            probs = torch.softmax(self._bias(self.model(batched)), dim=1)  # [K*D, C, size, size]
            flips = [torch.flip(probs[i * d:(i + 1) * d], dims) if dims else probs[i * d:(i + 1) * d]
                     for i, dims in enumerate(_FLIPS)]             # un-flip each block back
            members = torch.stack(flips)                          # [K, D, C, size, size]
            mean = members.mean(0)                                # [D, C, size, size] mean softmax
            pred = mean.argmax(1).to(torch.uint8).cpu().numpy()
        return pred, mean, members

    def predict_volume_probs(self, vol_img: Float[np.ndarray, "d h w"]):
        """Mean-softmax over the 4 TTA flips — the shared inference primitive. Returns
        (pred uint8 [D,size,size], mean_softmax [D,C,size,size]). See predict_volume_members for the
        per-member stack used by the uncertainty decomposition."""
        pred, mean, _ = self.predict_volume_members(vol_img)
        return pred, mean

    @shapecheck
    def predict_volume(self, vol_img: Float[np.ndarray, "d h w"], *, tta: bool = False) -> UInt8[np.ndarray, "d s s"]:
        """Predict a label map [D, size, size] for one z-scored [D, H, W] volume.

        `tta=True` averages over the 4 in-plane flips (delegates to predict_volume_probs); `tta=False`
        is a single batched forward. argmax of the flip-sum == argmax of the mean, so results match."""
        if tta:
            pred, _ = self.predict_volume_probs(vol_img)
            return pred
        self.model.eval()
        xs = self._stack_slices(vol_img)
        with torch.no_grad():
            x = torch.from_numpy(xs)[:, None].to(self.device)     # [D, 1, size, size]
            return self._bias(self.model(x)).argmax(1).cpu().numpy().astype(np.uint8)
