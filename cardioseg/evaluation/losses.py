"""Segmentation losses, built from a LossCfg (cardioseg.hparams).

Baseline `dice_ce` (MONAI Dice+CE) is the cardiac standard: CE gives stable gradients, Dice handles
the heavy bg/fg imbalance. Both are *region* losses — blind to a thin boundary over-fill (the ES
cavity over-segmentation that drives the EF bias). `dice_ce_hd` adds a Hausdorff-DT boundary term,
ramped in over a warmup (HD losses diverge if applied from epoch 0), to attack that directly.
"""
from cardioseg.hparams import LossCfg


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


def build_loss(cfg: LossCfg | None = None):
    """Loss callable from a LossCfg. Returns an object with __call__(logits, y); for dice_ce_hd it
    also has an `.epoch` attribute the train loop updates to drive the HD warmup ramp."""
    cfg = cfg or LossCfg()
    if cfg.kind == "dice_ce":
        return dice_ce_loss()
    if cfg.kind == "dice_ce_hd":
        return DiceCEHD(hd_weight=cfg.hd_weight, warmup=cfg.hd_warmup, ramp=cfg.hd_ramp)
    raise ValueError(f"unknown loss kind {cfg.kind!r} (dice_ce | dice_ce_hd)")
