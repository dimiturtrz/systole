"""Segmentation losses, built from a LossCfg (core.hparams).

Baseline `dice_ce` (MONAI Dice+CE) is the cardiac standard: CE gives stable gradients, Dice handles
the heavy bg/fg imbalance. Both are *region* losses — blind to a thin boundary over-fill (the ES
cavity over-segmentation that drives the EF bias). `dice_ce_hd` adds a Hausdorff-DT boundary term,
ramped in over a warmup (HD losses diverge if applied from epoch 0), to attack that directly.
"""
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss, DiceLoss, HausdorffDTLoss, TverskyLoss

from core.hparams import LossCfg


def dice_ce_loss(*, to_onehot_y=True, softmax=True, **kw):
    """MONAI compound Dice + CE for integer label maps (0/1/2/3)."""
    return DiceCELoss(to_onehot_y=to_onehot_y, softmax=softmax, **kw)


class DiceCEHD:
    """Dice+CE, then + λ·Hausdorff-DT phased in AFTER a pure-Dice warmup.

    HD-DT is ~50x the Dice scale and expensive (distance transforms), and diverges if applied from
    epoch 0. So: epochs < warmup -> pure Dice+CE, HD not even computed (fast); then HD ramps 0->λ
    over `ramp` epochs while masks are already roughly right. The train loop sets `.epoch` each epoch.
    """

    def __init__(self, hd_weight: float = 0.01, warmup: int = 15, ramp: int = 5):
        self.dce = DiceCELoss(to_onehot_y=True, softmax=True)
        self.hd = HausdorffDTLoss(to_onehot_y=True, softmax=True, include_background=False)
        self.hd_weight, self.warmup, self.ramp, self.epoch = hd_weight, warmup, ramp, 0

    def __call__(self, logits, y):
        loss = self.dce(logits, y)
        if self.hd_weight > 0 and self.epoch >= self.warmup:        # skip HD compute during warmup
            ramp = min(1.0, (self.epoch - self.warmup + 1) / max(1, self.ramp))
            loss = loss + self.hd_weight * ramp * self.hd(logits, y)
        return loss


class DiceCETversky:
    """Dice+CE + λ·Tversky(α,β). With β>α the Tversky term penalizes false positives harder, directly
    discouraging over-segmentation (the ES cavity over-fill). Pure GPU region loss — stable from
    epoch 0, no warmup, as fast as Dice (unlike Hausdorff-DT, which is CPU-bound on Windows)."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, lam: float = 1.0):
        self.dce = DiceCELoss(to_onehot_y=True, softmax=True)
        self.tv = TverskyLoss(to_onehot_y=True, softmax=True, alpha=alpha, beta=beta)
        self.lam = lam

    def __call__(self, logits, y):
        return self.dce(logits, y) + self.lam * self.tv(logits, y)


class HausdorffERLoss:
    """Erosion-based Hausdorff surrogate (Karimi & Salcudean 2019, method 2) — pure-torch GPU,
    differentiable, no distance transform (so no cucim, unlike HD-DT).

    Algorithm per the reference (PatRyg99/HausdorffLoss): error = (prob - onehot)^2, then iterate
    {conv-spread (cross kernel) -> threshold (relu(·-0.5)) -> min-max renormalize}, accumulating each
    eroded level weighted by (k+1)^alpha. Deep-interior (far-from-boundary) errors survive more
    erosions -> heavier weight -> Hausdorff-like (penalize the farthest errors hardest). The reference
    runs scipy/CPU under @no_grad (untrainable); this is on-device torch with grad through the error
    term -> a real trainable GPU boundary loss. Best for stray voxels / loose boundaries (RV), less so
    for thin near-boundary over-fill (use Tversky for that).
    """

    def __init__(self, alpha: float = 2.0, erosions: int = 10, *, include_background: bool = False):
        self.alpha, self.erosions, self.include_background = alpha, erosions, include_background
        self._cross = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]]) * 0.2  # sum=1

    def __call__(self, logits, y):
        # Force fp32 for the whole loss: under AMP autocast, F.conv2d would re-cast to fp16 and the
        # min-max renormalize's 1e-8 clamp underflows to 0 -> divide-by-zero -> NaN.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            prob = logits.float().softmax(1)
            c = prob.shape[1]
            tgt = y[:, 0] if y.dim() == 4 else y                  # noqa: PLR2004  [B,H,W]
            onehot = F.one_hot(tgt.long(), c).permute(0, 3, 1, 2).float()
            if not self.include_background:
                prob, onehot = prob[:, 1:], onehot[:, 1:]
            ch = prob.shape[1]
            err = (prob - onehot) ** 2                            # [B,ch,H,W], grad -> prob
            k2 = self._cross.to(prob.device, torch.float32).view(1, 1, 3, 3).expand(ch, 1, 3, 3)
            # distance weight via erosion — DETACHED (like the DT in HD-DT). Gradient flows only
            # through `err`; the erosion/min-max chain would otherwise blow gradients up (÷ tiny -> NaN).
            with torch.no_grad():
                b = err.detach().clone()
                w = torch.zeros_like(b)
                for k in range(self.erosions):
                    b = F.relu(F.conv2d(b, k2, padding=1, groups=ch) - 0.5)
                    flat = b.flatten(2)
                    mn = flat.min(-1, keepdim=True).values
                    rng = (flat.max(-1, keepdim=True).values - mn).clamp_min(1e-8)
                    b = ((flat - mn) / rng).view_as(b)
                    w = w + b * float((k + 1) ** self.alpha)
            return (err * w).mean()                               # error weighted by distance-to-boundary


class DiceCEHER:
    """Dice+CE + λ·Hausdorff-ER, HER ramped in after a pure-Dice warmup (same schedule as DiceCEHD,
    but HER is GPU-cheap)."""

    def __init__(self, her_weight=0.5, alpha=2.0, erosions=10, warmup=5, ramp=5):
        self.dce = dice_ce_loss()
        self.her = HausdorffERLoss(alpha=alpha, erosions=erosions)
        self.her_weight, self.warmup, self.ramp, self.epoch = her_weight, warmup, ramp, 0

    def __call__(self, logits, y):
        loss = self.dce(logits, y)
        if self.her_weight > 0 and self.epoch >= self.warmup:
            r = min(1.0, (self.epoch - self.warmup + 1) / max(1, self.ramp))
            loss = loss + self.her_weight * r * self.her(logits, y)
        return loss


class SoftDiceCE:
    """Dice+CE for SOFT (probabilistic) targets [B, C, H, W] (channels sum to 1) — for soft-label
    training (boundary voxels are fractional). MONAI DiceCELoss can't be used: its CE term argmaxes
    the target, collapsing the soft labels back to hard. This keeps them:
      - soft Dice via MONAI DiceLoss(softmax=True, to_onehot_y=False) — already a soft-overlap loss
      - soft CE = -(target * log_softmax(logits)).sum(classes).mean()  — cross-entropy to a soft dist
    Reduces to the standard hard Dice+CE when the target is one-hot (sigma->0). fp32 under autocast
    (log_softmax + the channel sum are fp16-unsafe)."""

    def __init__(self, *, include_background: bool = True, lambda_dice: float = 1.0, lambda_ce: float = 1.0):
        self.dice = DiceLoss(softmax=True, to_onehot_y=False, include_background=include_background)
        self.ld, self.lce = lambda_dice, lambda_ce

    def __call__(self, logits, target):
        with torch.autocast(device_type=logits.device.type, enabled=False):
            logits, target = logits.float(), target.float()
            ce = -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            d = self.dice(logits, target)
        return self.ld * d + self.lce * ce


class PartialLabelDiceCE:
    """Dice+CE with a per-sample VALID-CLASS mask `valid` [B, C] bool — the general partial-label loss.

    A sample declares which classes' GT is trustworthy (e.g. SCD labels LV only; RV is unlabeled and
    lumped into background, so BOTH rv and bg are untrustworthy for it -> valid = {myo, cav}). Then:
      - Dice is averaged over each sample's VALID classes only (bg counts only where valid).
      - CE ignores every pixel whose GT (argmax) class is not valid for that sample — so SCD's
        RV-contaminated background contributes no gradient (never teaches 'RV -> bg').
    `valid=None` (or all-True) -> every class/pixel counts = ordinary Dice+CE. Accepts a hard target
    [B,1,H,W]/[B,H,W] or a soft one [B,C,H,W]. fp32 under autocast (softmax/log_softmax are fp16-unsafe).
    Used ONLY when a batch carries a mask; the full-label path stays on the existing (MONAI) losses.
    """

    def __init__(self, lambda_dice: float = 1.0, lambda_ce: float = 1.0, eps: float = 1e-5):
        self.ld, self.lce, self.eps = lambda_dice, lambda_ce, eps

    def __call__(self, logits, target, valid=None):
        B, C = logits.shape[:2]
        with torch.autocast(device_type=logits.device.type, enabled=False):
            logits = logits.float()
            if target.dim() == 4 and target.shape[1] == C:            # noqa: PLR2004  soft [B,C,H,W]
                oh = target.float()
                cls = oh.argmax(1)                                    # [B,H,W] dominant class (pixel validity)
            else:                                                     # hard labels
                cls = (target[:, 0] if target.dim() == 4 else target).long()  # noqa: PLR2004  rank check
                oh = F.one_hot(cls, C).permute(0, 3, 1, 2).float()
            if valid is None:
                valid = torch.ones(B, C, dtype=torch.bool, device=logits.device)
            vf = valid.float()
            prob = logits.softmax(1)
            inter = (prob * oh).sum((2, 3))                           # [B,C]
            denom = prob.sum((2, 3)) + oh.sum((2, 3))
            dice = (2 * inter + self.eps) / (denom + self.eps)        # [B,C]
            dice_loss = (1 - (dice * vf).sum(1) / vf.sum(1).clamp_min(1)).mean()
            ce_pix = -(oh * F.log_softmax(logits, 1)).sum(1)          # [B,H,W]
            valid_pix = valid.gather(1, cls.reshape(B, -1)).reshape(cls.shape).float()
            ce = (ce_pix * valid_pix).sum() / valid_pix.sum().clamp_min(1)
        return self.ld * dice_loss + self.lce * ce


def uncertainty_weighted(loss_terms, log_vars):
    """Kendall (2018) homoscedastic-uncertainty weighting: Σ exp(-s_i)·L_i + s_i, with s_i = log σ_i²
    LEARNABLE. The net auto-balances the tasks (lower s_i -> higher weight on L_i); the +s_i penalty
    stops s_i -> ∞ (which would zero every weight). Retires hand-set loss weights -> self-balancing,
    scale-free (paired with the dimensionless vol_loss). Pass the learnable log-vars as parameters in
    the optimizer."""
    return sum(torch.exp(-s) * l + s for l, s in zip(loss_terms, log_vars, strict=True))


def build_loss(cfg: LossCfg | None = None):
    """Loss callable from a LossCfg. Returns an object with __call__(logits, y); dice_ce_hd also has
    an `.epoch` attribute the train loop updates to drive the HD warmup ramp."""
    cfg = cfg or LossCfg()
    if cfg.kind == "dice_ce":
        return dice_ce_loss()
    if cfg.kind == "dice_ce_tversky":
        return DiceCETversky(cfg.tversky_alpha, cfg.tversky_beta, cfg.tversky_lambda)
    if cfg.kind == "dice_ce_her":
        return DiceCEHER(cfg.her_weight, cfg.her_alpha, cfg.her_erosions, cfg.her_warmup, cfg.her_ramp)
    if cfg.kind == "dice_ce_hd":
        return DiceCEHD(hd_weight=cfg.hd_weight, warmup=cfg.hd_warmup, ramp=cfg.hd_ramp)
    raise ValueError(f"unknown loss {cfg.kind!r} (dice_ce | dice_ce_tversky | dice_ce_her | dice_ce_hd)")
