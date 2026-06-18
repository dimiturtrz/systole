"""Segmentation losses. Compound Dice + cross-entropy is the cardiac standard:
CE gives stable gradients, Dice handles the heavy background/foreground imbalance
(bg dominates the short-axis frame). nnU-Net and most ACDC methods use this combo.

Thin wrappers over MONAI so the training code imports losses from one place and we
can swap implementations without touching the loop.
"""


def dice_ce_loss(to_onehot_y=True, softmax=True, **kw):
    """Compound Dice + CE loss for multi-class masks (labels 0/1/2/3).

    to_onehot_y: targets are integer label maps (not one-hot).
    softmax: apply softmax to logits before the loss.
    """
    from monai.losses import DiceCELoss
    return DiceCELoss(to_onehot_y=to_onehot_y, softmax=softmax, **kw)
