"""Pure cores of the two figure/eval scripts that don't need a model:
  - overlay: mid-slice pick + the clean/HCM hero-case selection policy (compositing + savefig = shell)
  - soft_eval: the EDV/ESV -> EF scalar (the prediction loop + ECE need a model — shell)
"""
import numpy as np
import pytest

from cardioseg.evaluation.overlay import _mid_slice, pick_hero_cases
from cardioseg.evaluation.soft_eval import SoftEval


# --- _mid_slice: most-foreground slice ---
def test_mid_slice_picks_max_foreground():
    """Mid-ventricle = the slice with the most non-zero GT voxels."""
    vol = np.zeros((3, 4, 4), int)
    vol[1, :2, :2] = 1        # slice 1: 4 fg
    vol[2, 0, 0] = 1          # slice 2: 1 fg
    assert _mid_slice(vol) == 1


def test_mid_slice_all_empty_returns_zero():
    """Boundary: an all-background volume -> argmax of ties is index 0."""
    assert _mid_slice(np.zeros((3, 4, 4), int)) == 0


# --- pick_hero_cases: min clean-err + max HCM-err ---
def _c(name, group, ef_gt, ef_pred):
    return {"name": name, "group": group, "ef_gt": ef_gt, "ef_pred": ef_pred}


def test_pick_hero_selects_min_clean_and_max_hcm():
    """Policy class: clean row = smallest |ef_gt-ef_pred| among DCM/NOR/MINF; HCM row = largest err."""
    cases = [_c("a", "NOR", 60, 58), _c("b", "DCM", 40, 30),
             _c("c", "HCM", 70, 68), _c("d", "HCM", 70, 40)]
    clean, hcm = pick_hero_cases(cases)
    assert clean["name"] == "a" and hcm["name"] == "d"
    assert clean["ef_err"] == 2 and hcm["ef_err"] == 30


def test_pick_hero_raises_without_hcm():
    """Missing class: no HCM case -> max() over an empty generator raises ValueError."""
    with pytest.raises(ValueError, match="arg is an empty sequence|empty"):
        pick_hero_cases([_c("a", "NOR", 60, 58)])


# --- _ef: EDV/ESV -> EF scalar ---
def test_ef_normal():
    """Normal class: EF = (EDV-ESV)/EDV*100."""
    assert SoftEval._ef(100.0, 40.0) == 60.0


def test_ef_zero_edv_is_nan():
    """Collapse class: EDV<=0 (no cavity) -> NaN, never a divide-by-zero."""
    assert np.isnan(SoftEval._ef(0.0, 0.0))
