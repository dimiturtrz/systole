"""M&M-2 adapter pure logic (core.data.static.mri.mnm2) — the dataset-dir search, subject-code CSV
key transform, meta assembly (FIELD float-parse, fixed Spain country), and the SA resolve wiring.
NIfTI reads + CSV globbing are the integration shell."""
import nibabel as nib
import numpy as np

from core.data.static.mri.base import MNM_LABEL_MAP, load_frames
from core.data.static.mri.mnm2 import (
    Mnm2Adapter,
    _dataset_dir,
    load_ed_es,
    meta_from_info,
    mnm2_cases,
    mnm2_info,
)


# --- meta_from_info: disease/vendor passthrough, FIELD float, fixed country, unfilled demographics ---
def test_meta_from_info_fields():
    m = meta_from_info({"DISEASE": "HCM", "VENDOR": "Philips", "SCANNER": "Achieva", "FIELD": "3.0"})
    assert m["group"] == "HCM" and m["vendor"] == "Philips" and m["scanner"] == "Achieva"
    assert m["field_T"] == 3.0                                 # float-parsed
    assert m["country"] == "Spain" and m["centre"] is None     # fixed (paper), per-subject centre unknown
    assert m["age"] is None and m["sex"] is None               # not published


def test_meta_from_info_bad_field_none():
    """Non-numeric FIELD -> None (to_float), never a crash; empty dict all-None but country fixed."""
    assert meta_from_info({"FIELD": "n/a"})["field_T"] is None
    m = meta_from_info({})
    assert m["group"] is None and m["field_T"] is None and m["country"] == "Spain"


# --- _dataset_dir: finds a dir holding NNN/ subject folders across common nestings ---
def test_dataset_dir_direct_nnn(tmp_path, monkeypatch):
    (tmp_path / "001").mkdir()                                 # NNN dir directly in root
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path.parent))
    assert _dataset_dir(tmp_path) == tmp_path                  # "." sub matches


def test_dataset_dir_nested(tmp_path, monkeypatch):
    nested = tmp_path / "MnM2" / "dataset"; (nested / "007").mkdir(parents=True)
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path.parent))
    assert _dataset_dir(tmp_path) == nested                    # MnM2/dataset nesting resolved


def test_dataset_dir_env_override(tmp_path, monkeypatch):
    (tmp_path / "123").mkdir()
    monkeypatch.setenv("CARDIAC_MNM2_ROOT", str(tmp_path))
    assert _dataset_dir() == tmp_path                          # env root wins


# --- mnm2_cases: only 3-digit dirs, numerically sorted; non-NNN ignored ---
def test_mnm2_cases_filters_and_sorts(tmp_path, monkeypatch):
    for n in ("010", "002", "100", "abc"):
        (tmp_path / n).mkdir()
    monkeypatch.setenv("CARDIAC_MNM2_ROOT", str(tmp_path))
    cases = mnm2_cases()
    assert [c.name for c in cases] == ["002", "010", "100"]    # NNN only, sorted; 'abc' excluded


# --- load_ed_es resolve wiring: SA ED/ES paths, label remap applied ---
def test_load_ed_es_resolve_remaps(tmp_path):
    pid = "042"
    img = np.ones((2, 2, 2), np.float32)
    gt = np.zeros((2, 2, 2), np.float32); gt[0, 0, 0] = 1      # raw 1 = LV-cav
    for tag in ("ED", "ES"):
        for suf, arr in (("", img), ("_gt", gt)):
            im = nib.Nifti1Image(arr, np.eye(4)); im.header.set_zooms((1, 1, 1))
            nib.save(im, tmp_path / f"{pid}_SA_{tag}{suf}.nii.gz")

    def resolve(tag):
        return (tmp_path / f"{pid}_SA_{tag}.nii.gz", tmp_path / f"{pid}_SA_{tag}_gt.nii.gz", None)

    out = load_frames("HCM", resolve, MNM_LABEL_MAP)
    assert "ED" in out and "ES" in out and out["group"] == "HCM"
    assert 3 in np.unique(out["ED"]["gt"])                     # raw 1 -> canonical 3


# --- mnm2_info: subject-code zero-padded key from dataset_information.csv (sibling of dataset dir) ---
def test_mnm2_info_zero_pads_key(tmp_path, monkeypatch):
    ds = tmp_path / "dataset"; (ds / "007").mkdir(parents=True)
    (tmp_path / "dataset_information.csv").write_text("SUBJECT_CODE,DISEASE,VENDOR,FIELD\n7,HCM,GE,1.5\n")
    monkeypatch.setenv("CARDIAC_MNM2_ROOT", str(ds))
    info = mnm2_info()
    assert info["007"]["DISEASE"] == "HCM"                    # '7' -> '007' key transform


# --- _dataset_dir: no NNN dir anywhere -> falls back to the raw root (never crashes) ---
def test_dataset_dir_fallback_to_raw(tmp_path, monkeypatch):
    empty = tmp_path / "empty"; empty.mkdir()
    monkeypatch.setenv("CARDIAC_DATA", str(empty))            # raw = empty/raw, no NNN dirs
    assert _dataset_dir() == empty / "raw"                    # fallback = raw root


# --- load_ed_es: end-to-end SA ED/ES via nifti + CSV disease, canonical remap ---
def test_load_ed_es_end_to_end(tmp_path, monkeypatch):
    ds = tmp_path / "MnM2" / "dataset"; case = ds / "042"; case.mkdir(parents=True)
    img = np.ones((2, 2, 2), np.float32)
    gt = np.zeros((2, 2, 2), np.float32); gt[0, 0, 0] = 1; gt[1, 1, 1] = 3   # raw 1=LV-cav, 3=RV
    for tag in ("ED", "ES"):
        for suf, arr in (("", img), ("_gt", gt)):
            im = nib.Nifti1Image(arr, np.eye(4)); im.header.set_zooms((1, 1, 1))
            nib.save(im, case / f"042_SA_{tag}{suf}.nii.gz")
    (tmp_path / "MnM2" / "dataset_information.csv").write_text("SUBJECT_CODE,DISEASE\n42,HCM\n")
    out = load_ed_es(case)
    assert out["group"] == "HCM" and "ED" in out and "ES" in out
    assert 3 in np.unique(out["ED"]["gt"]) and 1 in np.unique(out["ED"]["gt"])   # raw 1->3, 3->1


# --- Mnm2Adapter: thin delegators ---
def test_mnm2_adapter_delegates(tmp_path, monkeypatch):
    ds = tmp_path / "MnM2" / "dataset"; (ds / "001").mkdir(parents=True)
    (tmp_path / "MnM2" / "dataset_information.csv").write_text("SUBJECT_CODE,DISEASE,VENDOR,FIELD\n1,NOR,GE,1.5\n")
    monkeypatch.setenv("CARDIAC_MNM2_ROOT", str(ds))
    a = Mnm2Adapter()
    assert a.name == "mnm2" and a.label_map == MNM_LABEL_MAP
    assert [c.name for c in a.cases()] == ["001"]             # delegates to mnm2_cases
    assert a.meta(ds / "001")["group"] == "NOR"              # delegates to meta_from_info
    led = a.load_ed_es(ds / "001")                           # delegates; no SA nifti -> no ED/ES frames
    assert "ED" not in led and "ES" not in led and led["group"] == "NOR"
