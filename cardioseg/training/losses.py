"""Segmentation losses, built from a LossCfg (core.hparams).

Baseline `dice_ce` (MONAI Dice+CE) is the cardiac standard: CE gives stable gradients, Dice handles
the heavy bg/fg imbalance. Both are *region* losses — blind to a thin boundary over-fill (the ES
cavity over-segmentation that drives the EF bias). `dice_ce_hd` adds a Hausdorff-DT boundary term,
ramped in over a warmup (HD losses diverge if applied from epoch 0), to attack that directly.
"""
from core.hparams import LossCfg


def dice_ce_loss(to_onehot_y=True, softmax=True, **kw):
    """MONAI compound Dice + CE for integer label maps (0/1/2/3)."""
    from monai.losses import DiceCELoss
    return DiceCELoss(to_onehot_y=to_onehot_y, softmax=softmax, **kw)


class DiceCEHD:
    """Dice+CE, then + λ·Hausdorff-DT phased in AFTER a pure-Dice warmup.

    HD-DT is ~50x the Dice scale and expensive (distance transforms), and diverges if applied from
    epoch 0. So: epochs < warmup -> pure Dice+CE, HD not even computed (fast); then HD ramps 0->λ
    over `ramp` epochs while masks are already roughly right. The train loop sets `.epoch` each epoch.
    """

    def __init__(self, hd_weight: float = 0.01, warmup: int = 15, ramp: int = 5):
        from monai.losses import DiceCELoss, HausdorffDTLoss
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
        from monai.losses import DiceCELoss, TverskyLoss
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

    def __init__(self, alpha: float = 2.0, erosions: int = 10, include_background: bool = False):
        import torch
        self.alpha, self.erosions, self.include_background = alpha, erosions, include_background
        self._cross = torch.tensor([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]]) * 0.2  # sum=1

    def __call__(self, logits, y):
        import torch
        import torch.nn.functional as F
        # Force fp32 for the whole loss: under AMP autocast, F.conv2d would re-cast to fp16 and the
        # min-max renormalize's 1e-8 clamp underflows to 0 -> divide-by-zero -> NaN.
        with torch.autocast(device_type=logits.device.type, enabled=False):
            prob = logits.float().softmax(1)
            c = prob.shape[1]
            tgt = y[:, 0] if y.dim() == 4 else y                  # [B,H,W]
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
