"""Auxiliary EF/volume objectives — the lanes added on top of the dense seg loss.

Both lanes share one shape: fit-square'd slice stacks built ONCE, GPU-resident, then per epoch a
single BATCHED forward + `index_add_` segment-sum back to per-subject soft LV-cav volume. (The naive
version reloaded npz/cine from disk and forwarded one subject at a time every epoch — 3s -> 9.5s per
epoch for a 2-scalar signal.) instance-norm (the default UNet) is per-sample, so a batched forward is
numerically identical to the per-subject loop it replaces.

Design: each lane is an `AuxObjective` with a single `.loss(model) -> Tensor | None`. The train loop
holds a `list[AuxObjective]` and never branches on *which* lanes are active — `build_aux(cfg, …)`
assembles the active ones. Adding a lane = one class + one registry line, no new `if` in the loop.
  - VolConsistency  labeled subjects, GT EDV/ESV targets  -> dimensionless vol_loss.
  - KaggleEF        EF-only cine (no masks), EF-RATIO target (spacing-invariant, sidesteps Kaggle's
                    ambiguous slice-spacing). Bounded GPU pool (`pool`).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from core.data.static.labels import LV_CAV
from core.data.static.mri.kaggle_dsb import kaggle_cases, kaggle_ef, load_sax
from core.data.static.store import load_arrays
from core.measure import label_volume_ml, voxel_volume_ml
from core.preprocessing.preprocess import fit_square

from .volumes import vol_loss


# ---- shared primitives ------------------------------------------------------------------------
def _stack(vol, size: int, device: str) -> torch.Tensor:
    """[D,H,W] numpy -> [D,1,size,size] float32 on device (grid-fit, no augmentation)."""
    slices = [torch.from_numpy(fit_square(vol[z], size, 0.0)) for z in range(vol.shape[0])]
    return torch.stack(slices)[:, None].to(device)


def _zscore(s):
    s = s.astype(np.float32)
    return (s - s.mean()) / (s.std() + 1e-6)


def ef_ratio(ed: torch.Tensor, es: torch.Tensor) -> torch.Tensor:
    """EF% from per-subject ED/ES cavity totals (any consistent unit — px or mL): (ED-ES)/ED×100. The
    spacing cancels (a ratio), so this works straight on pixel-count vols. ED clamped at 1e-6 so an
    all-empty prediction gives 0/ε≈0% not a NaN. Scalar or [K]-vector; the pure core of KaggleEF.loss."""
    return (ed - es) / ed.clamp_min(1e-6) * 100.0


def ef_ratio_loss(ed: torch.Tensor, es: torch.Tensor, targets, delta: float = 0.1) -> torch.Tensor:
    """Huber loss of predicted EF-ratio vs csv EF targets, both in [0,1] (÷100). Dimensionless, spacing-
    invariant — the KaggleEF objective. `targets` in percent; a [K] tensor/list. Pulled out of .loss so
    the weak-supervision math is testable with synthetic cavity totals (no cine, no GPU forward)."""
    ef_pred = ef_ratio(ed, es)
    tgt = ed.new_tensor([float(t) for t in targets]) if not isinstance(targets, torch.Tensor) else targets
    return F.huber_loss(ef_pred / 100, tgt.to(ef_pred) / 100, delta=delta)


def _cav_volume(model, stacks, sizes, lv: int, *, amp: bool) -> torch.Tensor:
    """One batched forward over `stacks` [ΣDi,1,H,W]; soft LV-cav pixel-count per slice, segment-summed
    by the per-item slice-counts `sizes` -> per-item cavity pixel totals [K] (fp32, grad-carrying)."""
    owner = torch.repeat_interleave(torch.arange(len(sizes), device=stacks.device), sizes)
    with torch.autocast("cuda", enabled=amp):
        pix = model(stacks).softmax(1)[:, lv].float().sum((1, 2))       # [ΣDi]
    return torch.zeros(len(sizes), device=pix.device, dtype=pix.dtype).index_add_(0, owner, pix)


# ---- lanes: each just needs `.loss(model, amp) -> Tensor | None`; the loop iterates a list of them.
#      no Protocol/ABC ceremony — python duck-types, and `build_aux` is the single home for the contract.
class VolConsistency:
    """Labeled ED/ES stacks resident on GPU + GT EDV/ESV. Each epoch sample `k` subjects, one batched
    forward per phase, segment-sum -> dimensionless vol_loss vs the GT volumes. Subjects missing ED/ES
    or with no cavity are skipped at build."""

    def __init__(self, npz_paths, size: int, device: str, k: int, lv_label: int = LV_CAV):
        self.device, self.lv, self.k = device, lv_label, k
        self.ed, self.es = [], []                       # per-subject [Di,1,H,W] GPU stacks (aligned)
        edv_gt, esv_gt, vox = [], [], []
        for p in npz_paths:
            case = load_arrays(p)
            if "ed_img" not in case or "es_img" not in case:
                continue
            spacing = tuple(float(s) for s in case["spacing"])
            edv = label_volume_ml(case["ed_gt"], lv_label, spacing)
            esv = label_volume_ml(case["es_gt"], lv_label, spacing)
            if edv <= 0:
                continue
            self.ed.append(_stack(case["ed_img"], size, device))
            self.es.append(_stack(case["es_img"], size, device))
            edv_gt.append(edv); esv_gt.append(esv); vox.append(voxel_volume_ml(spacing))
        self.n = len(self.ed)
        self.counts = torch.tensor([t.shape[0] for t in self.ed], device=device)
        self.vox = torch.tensor(vox, device=device)
        self.edv_gt = torch.tensor(edv_gt, device=device)
        self.esv_gt = torch.tensor(esv_gt, device=device)

    def loss(self, model, delta: float = 0.1, *, amp: bool = True):
        if self.n == 0:
            return None
        idx = torch.randperm(self.n, device=self.device)[:self.k]
        sizes, vox = self.counts[idx], self.vox[idx]
        ed = torch.cat([self.ed[int(i)] for i in idx])                  # [ΣDi,1,H,W]
        es = torch.cat([self.es[int(i)] for i in idx])
        edv = _cav_volume(model, ed, sizes, self.lv, amp=amp) * vox     # [K] soft EDV (mL)
        esv = _cav_volume(model, es, sizes, self.lv, amp=amp) * vox
        return vol_loss(edv, esv, self.edv_gt[idx], self.esv_gt[idx], delta)


class KaggleEF:
    """EF-only cine resident on GPU, bounded to `pool` cases (a full 1140-cine cache is ~tens of GB;
    a fixed random subset is plenty weak-supervision). Each case -> one [L*P,1,H,W] tensor + (L,P) +
    EF%. `loss()` samples `k` cases and is fully batched: ONE no-grad forward over all sampled slices
    finds each case's ED/ES phase (max/min total cavity), then TWO grad forwards (all cases' ED, all
    cases' ES) segment-summed per case -> EF-RATIO Huber vs the csv EF. No dense mask, so the seg loss
    never touches these."""

    def __init__(self, cases, ef_targets: dict, size: int, device: str, k: int, pool: int = 96,  # noqa: PLR0913
                 lv_label: int = LV_CAV, seed: int = 0):
        self.device, self.lv, self.size, self.k = device, lv_label, size, k
        self.X, self.LP, self.ef = [], [], []          # [L*P,1,H,W] GPU / (L,P) / EF%
        for j in np.random.RandomState(seed).permutation(len(cases)):
            if len(self.X) >= pool:
                break
            c = cases[j]
            t = ef_targets.get(c.name)
            if not t or not t.get("ef"):
                continue
            sax = load_sax(c)
            if not sax:
                continue
            P, L = min(v.shape[0] for v, _, _ in sax), len(sax)
            arr = np.array([[fit_square(_zscore(vol[p]), size, 0.0) for p in range(P)]
                            for vol, _, _ in sax])                      # [L,P,H,W]
            # Kept on CPU (host RAM): the pool is large and each cine is touched rarely (k sampled/
            # epoch). Residing it in VRAM hoards ~tens of GB and thrashes the card — the big rarely-hit
            # pool belongs in RAM, sampled cines ship to the GPU per epoch (mirrors residency='cpu').
            self.X.append(torch.from_numpy(arr).reshape(L * P, 1, size, size))   # CPU tensor
            self.LP.append((L, P))
            self.ef.append(float(t["ef"]))
        self.n = len(self.X)

    def loss(self, model, delta: float = 0.1, *, amp: bool = True):
        if self.n == 0:
            return None
        idx = [int(i) for i in torch.randperm(self.n)[:self.k]]
        xs = [self.X[i].to(self.device, non_blocking=True) for i in idx]   # ship sampled cines to GPU
        sizes = [x.shape[0] for x in xs]
        with torch.no_grad(), torch.autocast("cuda", enabled=amp):      # phase-find, batched
            pix = model(torch.cat(xs)).softmax(1)[:, self.lv].float().sum((1, 2))
        ed_stacks, es_stacks, ls, tgt = [], [], [], []
        off = 0
        for gx, i, sz in zip(xs, idx, sizes, strict=True):                            # gx = the GPU-resident cine
            L, P = self.LP[i]
            pv = pix[off:off + sz].view(L, P).sum(0); off += sz          # [P] cavity vol / phase
            X = gx.view(L, P, 1, self.size, self.size)
            ed_stacks.append(X[:, int(pv.argmax())])
            es_stacks.append(X[:, int(pv.argmin())])
            ls.append(L); tgt.append(self.ef[i])
        ls = torch.tensor(ls, device=self.device)
        ed = _cav_volume(model, torch.cat(ed_stacks), ls, self.lv, amp=amp)  # [K] ED cavity vol (px)
        es = _cav_volume(model, torch.cat(es_stacks), ls, self.lv, amp=amp)
        return ef_ratio_loss(ed, es, tgt, delta)                         # spacing-cancelling EF Huber


def build_aux(cfg, splits, train_df, device: str, *, is_static: bool) -> list:
    """Assemble the active auxiliary lanes from a TrainCfg. Empty list when the EF lane is off or the
    train source isn't static (EDV/ESV need labeled patient frames). The train loop iterates the list;
    it never inspects cfg.ef_* itself. `size` is config (cfg.generator.data.size), not an arg."""
    if cfg.ef_lambda <= 0 or not is_static:
        return []
    size = cfg.generator.data.size
    lanes: list = [VolConsistency(splits.paths(train_df), size, device, cfg.ef_subjects)]
    if cfg.ef_kaggle:
        lanes.append(KaggleEF(kaggle_cases("train"), kaggle_ef("train"), size, device,
                              cfg.ef_kaggle_subjects, seed=cfg.seed))
    return lanes
