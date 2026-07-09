"""Canonical-label SSOT tests: the derived sets must stay consistent with CLASSES + the enum."""
import numpy as np

from core.data.static.labels import CLASS_NAMES, CLASSES, FOREGROUND, LV_CAV, Label, Labels


def test_enum_values():
    """The convention is fixed: bg/RV/myo/LV-cav = 0/1/2/3."""
    assert (Label.BG, Label.RV, Label.MYO, Label.LV_CAV) == (0, 1, 2, 3)


def test_derived_sets_consistent_with_classes():
    """FOREGROUND, CLASS_NAMES, LV_CAV are all derived from CLASSES — no independent literals."""
    assert FOREGROUND == tuple(CLASSES) == (1, 2, 3)        # bg excluded, label order
    assert CLASS_NAMES == [n for n, _ in CLASSES.values()] == ["RV", "LV-myo", "LV-cav"]
    assert LV_CAV == int(Label.LV_CAV) == 3
    assert Label.BG not in FOREGROUND                       # background never a foreground class


def test_overlay_cmap_bg_transparent_and_tracks_classes():
    """4 colors (bg + 3 classes); bg fully transparent; fg colors come from CLASSES hex."""
    from matplotlib.colors import to_rgb
    cm = Labels.overlay_cmap(alpha=0.5)
    assert cm.N == 1 + len(FOREGROUND) == 4
    assert cm(0) == (0.0, 0.0, 0.0, 0.0)                    # label 0 transparent
    # label 1 (RV) matches CLASSES color at the requested alpha
    assert np.allclose(cm(1)[:3], to_rgb(CLASSES[1][1])) and cm(1)[3] == 0.5
