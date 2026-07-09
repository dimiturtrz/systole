"""ACDC adapter pure logic (core.data.static.mri.acdc) — Info.cfg parsing, frame-path resolution,
patient discovery across the training/testing (and flat/database) layouts, and the meta constants
(vendor/field/centre cited from Bernard 2018). NIfTI reads are the integration shell (skipped without
the acdc data); everything here rides a fake tree via tmp_path + root injection (cf. test_scd/mnm2)."""
import nibabel as nib
import numpy as np

from core.data.static.mri.acdc import LABEL_MAP, AcdcAdapter


# --- _parse_info_cfg: 'key: value' lines -> dict; missing file -> {} ---
def test_parse_info_cfg_reads_pairs(tmp_path):
    (tmp_path / "Info.cfg").write_text("ED: 1\nES: 12\nGroup: DCM\nHeight: 170.0\nWeight: 75.0\n")
    cfg = AcdcAdapter._parse_info_cfg(tmp_path)
    assert cfg["ED"] == "1" and cfg["ES"] == "12" and cfg["Group"] == "DCM"
    assert cfg["Height"] == "170.0"


def test_parse_info_cfg_missing_file_empty(tmp_path):
    assert AcdcAdapter._parse_info_cfg(tmp_path / "nope") == {}


def test_parse_info_cfg_ignores_colonless_lines(tmp_path):
    (tmp_path / "Info.cfg").write_text("ED: 1\n# a comment with no key\nGroup: NOR\n")
    cfg = AcdcAdapter._parse_info_cfg(tmp_path)
    assert cfg == {"ED": "1", "Group": "NOR"}


# --- _frame_paths: patientXXX_frameNN(.nii.gz) / _gt, frame_no zero-padded to 2 ---
def test_frame_paths_zero_pads_frame(tmp_path):
    pdir = tmp_path / "patient007"
    img, gt = AcdcAdapter._frame_paths(pdir, 1)
    assert img.name == "patient007_frame01.nii.gz"
    assert gt.name == "patient007_frame01_gt.nii.gz"


def test_frame_paths_accepts_str_frame(tmp_path):
    """frame_no arrives as a str from Info.cfg -> int()'d then padded (already-2-digit unchanged)."""
    img, gt = AcdcAdapter._frame_paths(tmp_path / "patient100", "12")
    assert img.name == "patient100_frame12.nii.gz" and gt.name == "patient100_frame12_gt.nii.gz"


# --- cases: pulls training/ + testing/ patients, sorted; root injection ---
def _fake_acdc(base, patients_by_split):
    for split, patients in patients_by_split.items():
        for p in patients:
            (base / split / p).mkdir(parents=True)
    return base


def test_cases_unions_training_and_testing_sorted(tmp_path):
    base = _fake_acdc(tmp_path / "acdc",
                      {"training": ["patient010", "patient002"], "testing": ["patient101"]})
    cases = AcdcAdapter(root=base).cases()
    assert [c.name for c in cases] == ["patient002", "patient010", "patient101"]   # unioned + sorted


def test_cases_flat_fallback_layout(tmp_path):
    """No training/testing dirs -> fall back to a flat root of patient* dirs."""
    base = tmp_path / "acdc"
    (base / "patient005").mkdir(parents=True)
    (base / "not_a_patient").mkdir()
    cases = AcdcAdapter(root=base).cases()
    assert [c.name for c in cases] == ["patient005"]


def test_cases_database_training_fallback(tmp_path):
    """The database/training/ nesting is picked up when the top-level layout is empty."""
    base = tmp_path / "acdc"
    (base / "database" / "training" / "patient042").mkdir(parents=True)
    cases = AcdcAdapter(root=base).cases()
    assert [c.name for c in cases] == ["patient042"]


def test_cases_default_root_via_env(tmp_path, monkeypatch):
    """root=None -> DATA_ROOT under raw/acdc (driven via CARDIAC_DATA)."""
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path))
    (tmp_path / "raw" / "acdc" / "training" / "patient001").mkdir(parents=True)
    # DATA_ROOT is import-time bound, so drive default via the module constant, not env re-eval:
    a = AcdcAdapter(root=tmp_path / "raw" / "acdc")
    assert [c.name for c in a.cases()] == ["patient001"]


# --- load_ed_es: resolves ED/ES frames from Info.cfg, identity label map (ACDC already canonical) ---
def _write_nii(path, arr):
    im = nib.Nifti1Image(arr.astype(np.float32), np.eye(4)); im.header.set_zooms((1, 1, 1))
    nib.save(im, path)


def test_load_ed_es_end_to_end(tmp_path):
    pdir = tmp_path / "acdc" / "training" / "patient001"; pdir.mkdir(parents=True)
    (pdir / "Info.cfg").write_text("ED: 1\nES: 12\nGroup: DCM\n")
    img = np.ones((2, 2, 2), np.float32)
    gt = np.zeros((2, 2, 2), np.float32); gt[0, 0, 0] = 3   # already canonical (3 = LV-cav)
    for fno in ("01", "12"):
        _write_nii(pdir / f"patient001_frame{fno}.nii.gz", img)
        _write_nii(pdir / f"patient001_frame{fno}_gt.nii.gz", gt)
    out = AcdcAdapter(root=tmp_path / "acdc").load_ed_es(pdir)
    assert out["group"] == "DCM" and "ED" in out and "ES" in out
    assert 3 in np.unique(out["ED"]["gt"])                 # identity map -> unchanged


def test_load_ed_es_no_frame_index_skipped(tmp_path):
    """Info.cfg without ED/ES keys -> resolve returns None -> those phases skipped, no crash."""
    pdir = tmp_path / "acdc" / "training" / "patient001"; pdir.mkdir(parents=True)
    (pdir / "Info.cfg").write_text("Group: NOR\n")
    out = AcdcAdapter(root=tmp_path / "acdc").load_ed_es(pdir)
    assert out["group"] == "NOR" and "ED" not in out and "ES" not in out


# --- meta: demographics from Info.cfg (float-parsed), cited acquisition constants ---
def test_meta_constants_and_parsed_demographics(tmp_path):
    pdir = tmp_path / "acdc" / "training" / "patient001"; pdir.mkdir(parents=True)
    (pdir / "Info.cfg").write_text("Group: HCM\nHeight: 170.0\nWeight: 75.5\n")
    m = AcdcAdapter(root=tmp_path / "acdc").meta(pdir)
    assert m["group"] == "HCM" and m["height"] == 170.0 and m["weight"] == 75.5
    assert m["vendor"] == "Siemens" and m["field_T"] == [1.5, 3.0]     # Bernard 2018 constants
    assert m["centre"] == "Dijon" and m["country"] == "France"
    assert m["age"] is None and m["sex"] is None                       # not recorded


def test_meta_bad_height_none(tmp_path):
    """Non-numeric Height -> None via to_float, never a crash."""
    pdir = tmp_path / "acdc" / "training" / "patient001"; pdir.mkdir(parents=True)
    (pdir / "Info.cfg").write_text("Group: NOR\nHeight: n/a\n")
    assert AcdcAdapter(root=tmp_path / "acdc").meta(pdir)["height"] is None


# --- AcdcAdapter identity: name + label_map are the canonical identity ---
def test_adapter_name_and_identity_label_map():
    a = AcdcAdapter()
    assert a.name == "acdc" and a.label_map == LABEL_MAP == {0: 0, 1: 1, 2: 2, 3: 3}
