"""Kaggle DSB reader (core.data.static.mri.kaggle_dsb) — EF/regression, not seg. The pure core is the EF
computation; the DICOM/CSV paths are integration (skipped without the data)."""
import pytest

from core.data.static.mri.kaggle_dsb import _ef, kaggle_cases, kaggle_ef


def test_ef_computation():
    e = _ef(246.7, 108.3)                                    # EDV, ESV (mL)
    assert e["edv"] == 246.7 and e["esv"] == 108.3
    assert abs(e["ef"] - 56.1) < 0.05                        # (EDV-ESV)/EDV * 100
    assert _ef(0, 0)["ef"] is None                           # guard div-by-zero


def test_targets_and_cases_if_data_present():
    try:
        ef, cases = kaggle_ef("train"), kaggle_cases("train")
    except (FileNotFoundError, OSError):
        pytest.skip("kaggle_dsb2015 not present")
    if not ef:
        pytest.skip("no targets")
    cid, t = next(iter(ef.items()))
    assert {"edv", "esv", "ef"} <= t.keys() and 0 <= t["ef"] <= 100
