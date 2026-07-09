"""Kaggle DSB reader (core.data.static.mri.kaggle_dsb) — EF/regression, not seg. The pure core is the EF
computation + CSV target parsing; the DICOM image reads are integration (skipped without the data)."""
import polars as pl
import pytest

from core.data.static.mri.kaggle_dsb import (
    COUNTRY,
    _base,
    _ef,
    _split_dir,
    build_kaggle_meta,
    kaggle_cases,
    kaggle_ef,
    kaggle_meta,
)


def test_ef_computation():
    e = _ef(246.7, 108.3)                                    # EDV, ESV (mL)
    assert e["edv"] == 246.7 and e["esv"] == 108.3
    assert abs(e["ef"] - 56.1) < 0.05                        # (EDV-ESV)/EDV * 100
    assert _ef(0, 0)["ef"] is None                           # guard div-by-zero


# --- _base / _split_dir: nested <split>/<split>/ layout + root override ---
def test_split_dir_nested(tmp_path):
    assert _split_dir("train", tmp_path) == tmp_path / "train" / "train"   # nested split dir
    assert _base(tmp_path) == tmp_path                        # explicit root passthrough


# --- kaggle_ef: train/validate CSV (Diastole/Systole cols) ---
def test_kaggle_ef_train_csv(tmp_path):
    (tmp_path / "train.csv").write_text("Id,Diastole,Systole\n1,200,80\n2,150,150\n")
    ef = kaggle_ef("train", tmp_path)
    assert ef["1"]["edv"] == 200 and ef["1"]["esv"] == 80
    assert abs(ef["1"]["ef"] - 60.0) < 0.01                   # (200-80)/200
    assert ef["2"]["ef"] == 0.0                               # EDV==ESV -> 0% (still computed)


# --- kaggle_ef: test path via solution.csv (two rows per case, phase suffix) ---
def test_kaggle_ef_test_solution_csv(tmp_path):
    (tmp_path / "solution.csv").write_text(
        "Id,Volume,Usage\nc7_Diastole,200,Public\nc7_Systole,80,Public\n"
        "c8_Diastole,150,Public\n")                          # c8 missing Systole -> dropped
    ef = kaggle_ef("test", tmp_path)
    assert set(ef) == {"c7"}                                  # only the complete pair kept
    assert ef["c7"]["edv"] == 200 and abs(ef["c7"]["ef"] - 60.0) < 0.01


# --- kaggle_cases: numerically sorted case dirs under the nested split ---
def test_kaggle_cases_numeric_sort(tmp_path):
    d = tmp_path / "train" / "train"
    for n in ("10", "2", "1"):
        (d / n).mkdir(parents=True)
    assert [c.name for c in kaggle_cases("train", tmp_path)] == ["1", "2", "10"]   # numeric, not lexical


# --- kaggle_meta: location constants + EF-target merge; no SAX dir -> DICOM read skipped ---
def test_kaggle_meta_no_sax_location_and_ef(tmp_path):
    """No sax_* series -> DICOM block skipped; location constants + the case's EF target still returned."""
    case = tmp_path / "17"; (case / "study").mkdir(parents=True)   # empty study -> no sax_*
    m = kaggle_meta(case, ef_targets={"17": {"edv": 200, "esv": 80, "ef": 60.0}})
    assert m["country"] == COUNTRY and m["region"] == "North America"
    assert m["edv"] == 200 and m["ef"] == 60.0                # EF target merged
    assert "vendor" not in m                                  # no DICOM -> acquisition fields absent


def test_kaggle_meta_no_target_match(tmp_path):
    """EF targets present but none for this case -> no edv/esv/ef keys (location only)."""
    case = tmp_path / "99"; (case / "study").mkdir(parents=True)
    m = kaggle_meta(case, ef_targets={"1": {"edv": 1, "esv": 1, "ef": 0.0}})
    assert "edv" not in m and m["country"] == COUNTRY


# --- build_kaggle_meta: writes processed/kaggle/<split>/meta.csv over cases (no DICOM in the tree) ---
def test_build_kaggle_meta_writes_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path))
    raw = tmp_path / "raw" / "kaggle_dsb2015"
    (raw / "train" / "train" / "5" / "study").mkdir(parents=True)   # study present, no sax_* -> no DICOM
    (raw / "train.csv").write_text("Id,Diastole,Systole\n5,200,80\n")
    out = build_kaggle_meta("train")
    assert out.name == "meta.csv" and out.parent.name == "train"
    df = pl.read_csv(out, infer_schema_length=0)             # all-Utf8: subject_id "5", not int
    assert df["subject_id"].to_list() == ["5"] and df["dataset"].to_list() == ["kaggle"]
    assert df["country"].to_list() == [COUNTRY] and abs(float(df["ef"][0]) - 60.0) < 0.01   # EF target carried


def test_targets_and_cases_if_data_present():
    try:
        ef, cases = kaggle_ef("train"), kaggle_cases("train")
    except (FileNotFoundError, OSError):
        pytest.skip("kaggle_dsb2015 not present")
    if not ef:
        pytest.skip("no targets")
    assert all(c.is_dir() for c in cases)                     # case dirs exist on disk
    cid, t = next(iter(ef.items()))
    assert {"edv", "esv", "ef"} <= t.keys() and 0 <= t["ef"] <= 100
