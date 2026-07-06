"""Canonical segmentation label convention — the ONE source for the 0/1/2/3 scheme.

Verified on real masks (geometric myo-enclosure test): background 0, RV cavity 1,
LV myocardium 2, LV blood-pool cavity 3. LV is label 3, NOT 1 — always disambiguate
LV vs RV by this map, never by a remembered int elsewhere.

Everything that needs a label int, the foreground set, a class name, or a plot color
imports from here, so the convention can't drift across data adapters, training,
evaluation, and the model card.
"""
from __future__ import annotations

from enum import IntEnum


class Label(IntEnum):
    """The canonical label integers."""
    BG = 0       # background
    RV = 1       # right-ventricle cavity
    MYO = 2      # left-ventricle myocardium
    LV_CAV = 3   # left-ventricle blood-pool cavity


# Foreground class -> (display name, plot color). The single registry the eval +
# viewer + model card all read; ordered by label int.
CLASSES: dict[int, tuple[str, str]] = {
    int(Label.RV): ("RV", "#5b8def"),
    int(Label.MYO): ("LV-myo", "#ffca5b"),
    int(Label.LV_CAV): ("LV-cav", "#ef5350"),
}

FOREGROUND: tuple[int, ...] = tuple(CLASSES)        # (1, 2, 3) — non-background labels
LV_CAV: int = int(Label.LV_CAV)                     # the EF blood-pool label (3)
CLASS_NAMES: list[str] = [n for n, _ in CLASSES.values()]   # ["RV", "LV-myo", "LV-cav"]

# ── Partial-label: which classes a dataset actually ANNOTATES (trustworthy GT) ────────────────────
# Default = all four. SCD labels LV only (endo/epi contours) — RV is unlabeled and lumped into the
# background, so BOTH RV and bg are untrustworthy for it -> valid = {MYO, LV_CAV}. The partial-label
# loss (PartialLabelDiceCE) masks the rest so SCD never teaches "RV -> bg".
ALL_CLASSES: tuple[int, ...] = tuple(int(c) for c in Label)         # (0, 1, 2, 3)
LABELED_CLASSES: dict[str, tuple[int, ...]] = {
    "scd": (int(Label.MYO), int(Label.LV_CAV)),                     # LV-only; bg + RV untrusted
}


def valid_classes(dataset: str) -> tuple[int, ...]:
    """The classes whose GT is trustworthy for a dataset (default: all four)."""
    return LABELED_CLASSES.get(dataset, ALL_CLASSES)


def valid_row(dataset: str, n_classes: int = 4) -> list[bool]:
    """Per-class validity flags [C] for a dataset — one row of the [N, C] partial-label mask."""
    vc = set(valid_classes(dataset))
    return [c in vc for c in range(n_classes)]


def overlay_cmap(alpha: float = 0.5):
    """Matplotlib ListedColormap for a label overlay: background transparent, each
    foreground class its CLASSES color at `alpha`. Index i == label i (vmin=0, vmax=3)."""
    from matplotlib.colors import ListedColormap, to_rgb

    colors = [(0.0, 0.0, 0.0, 0.0)]  # label 0 = background, fully transparent
    for lab in FOREGROUND:
        colors.append((*to_rgb(CLASSES[lab][1]), alpha))
    return ListedColormap(colors)
