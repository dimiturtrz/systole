"""data/static/mri/base (class Base) shared-helper tests (equivalence classes): the primitives the
adapters reuse — to_float, load_csv_info, load_nifti, load_frames, apply_label_map, and the geometric
identify_lv_cavity."""
import numpy as np
import pytest

from core.data.static.mri.base import (
    LV_MYO,
    MNM_LABEL_MAP,
    Base,
    Dataset,
    Phase,
    Vendor,
)


# --- Dataset StrEnum: value == the processed-folder / V('dataset') / adapter.name string ---
def test_dataset_values_match_names():
    assert Dataset.ACDC == "acdc" and Dataset.CMRXMOTION == "cmrxmotion"   # drop-in for the name literals
    assert set(Dataset) >= {"acdc", "mnm2", "mnms1", "cmrxmotion", "scd"}  # the store datasets are members


# --- Vendor / Phase StrEnums: canonical value + case-insensitive _missing_ ---
def test_vendor_canonical_values_and_casefold():
    assert tuple(Vendor) == ("Siemens", "Philips", "GE", "Canon")   # canonical casing == the old literals
    assert Vendor.GE == "GE"                                        # StrEnum member IS its str value
    assert Vendor("siemens") is Vendor.SIEMENS                      # _missing_ folds a lowercase field
    assert Vendor("GE") is Vendor.GE                                # exact hit
    with pytest.raises(ValueError, match="Fuji"):
        Vendor("Fuji")                                             # unknown -> ValueError, not silent


def test_phase_canonical_values_and_casefold():
    assert tuple(Phase) == ("ED", "ES")
    assert Phase.ED == "ED" and Phase.ED.lower() == "ed"           # drop-in for the lowercase store keys
    assert Phase("ed") is Phase.ED and Phase("es") is Phase.ES      # the ED/ed case drift folds to one member
    assert Phase("ES") is Phase.ES
    with pytest.raises(ValueError, match="mid"):
        Phase("mid")


# --- to_float: parseable / None / garbage ---
def test_to_float_classes():
    assert Base.to_float("3.5") == 3.5            # parseable
    assert Base.to_float("7") == 7.0              # int-string
    assert Base.to_float(None) is None           # missing
    assert Base.to_float("n/a") is None          # garbage


# --- apply_label_map: identity (no-op) vs real remap ---
def test_apply_label_map_identity_unchanged():
    gt = np.array([[0, 1, 2, 3]], np.uint8)
    assert np.array_equal(Base.apply_label_map(gt, {0: 0, 1: 1, 2: 2, 3: 3}), gt)


def test_apply_label_map_mnm_flip():
    """M&Ms raw 1<->3 swap (LV-cav<->RV) reaches canonical."""
    gt = np.array([[0, 1, 2, 3]], np.uint8)
    out = Base.apply_label_map(gt, MNM_LABEL_MAP)
    assert list(out.ravel()) == [0, 3, 2, 1]
    assert MNM_LABEL_MAP == {0: 0, 1: 3, 2: 2, 3: 1}


# --- load_csv_info: key_col, alt fallback, transform, missing file ---
def _csv(tmp_path, text):
    p = tmp_path / "info.csv"
    p.write_text(text)
    return p


def test_load_csv_info_basic_and_transform(tmp_path):
    p = _csv(tmp_path, "SUBJECT_CODE,VENDOR\n7,Siemens\n12,GE\n")
    info = Base.load_csv_info(p, "SUBJECT_CODE", key_transform=lambda c: c.zfill(3))
    assert info["007"]["VENDOR"] == "Siemens" and info["012"]["VENDOR"] == "GE"


def test_load_csv_info_alt_key_fallback(tmp_path):
    """Falls back to alt_key_col when the primary column is absent (dual-spelling)."""
    p = _csv(tmp_path, "External_code,Vendor\nA1,Canon\n")
    info = Base.load_csv_info(p, "External code", alt_key_col="External_code")
    assert info["A1"]["Vendor"] == "Canon"


def test_load_csv_info_missing_file(tmp_path):
    assert Base.load_csv_info(tmp_path / "nope.csv", "k") == {}


# --- load_nifti: 3D transpose+spacing, 4D frame extraction ---
def test_load_nifti_3d_axes_and_spacing(tmp_path):
    nib = pytest.importorskip("nibabel")
    arr = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)  # (x, y, z)
    img = nib.Nifti1Image(arr, np.diag([1.1, 1.2, 5.0, 1.0]))     # zooms x,y,z
    img.header.set_zooms((1.1, 1.2, 5.0))
    p = tmp_path / "v.nii.gz"; nib.save(img, p)
    vol, sp = Base.load_nifti(p)
    assert vol.shape == (4, 3, 2)                 # x,y,z -> z,y,x = [D,H,W]
    assert sp == pytest.approx((5.0, 1.2, 1.1))   # (z, y, x)


def test_load_nifti_4d_frame(tmp_path):
    nib = pytest.importorskip("nibabel")
    arr = np.zeros((2, 2, 3, 4), np.float32)      # (x,y,z,t)
    arr[..., 2] = 9.0                              # mark frame 2
    img = nib.Nifti1Image(arr, np.eye(4)); img.header.set_zooms((1, 1, 1, 1))
    p = tmp_path / "cine.nii.gz"; nib.save(img, p)
    vol, _ = Base.load_nifti(p, frame=2)
    assert vol.shape == (3, 2, 2) and float(vol.max()) == 9.0   # picked frame 2


# --- load_frames template: resolve drives it; skip-None, missing-file skip, label-map applied ---
def _write_nii(tmp_path, name, arr):
    nib = pytest.importorskip("nibabel")
    img = nib.Nifti1Image(arr.astype(np.float32), np.eye(4)); img.header.set_zooms((1, 1, 1))
    p = tmp_path / name; nib.save(img, p)
    return p


def test_load_frames_resolve_and_label_map(tmp_path):
    img_p = _write_nii(tmp_path, "img.nii.gz", np.ones((2, 2, 2)))
    gt_p = _write_nii(tmp_path, "gt.nii.gz", np.array([[[0, 1]], [[2, 3]]]))  # raw labels
    # resolve: ED -> the files; ES -> None (skip)
    out = Base.load_frames("DCM", lambda tag: (img_p, gt_p, None) if tag == "ED" else None, MNM_LABEL_MAP)
    assert out["group"] == "DCM" and "ES" not in out and out["spacing"] is not None
    assert sorted(int(x) for x in np.unique(out["ED"]["gt"])) == [0, 1, 2, 3]   # remapped (1<->3)


def test_load_frames_missing_file_skipped(tmp_path):
    """A resolve pointing at a non-existent image is skipped, not an error."""
    out = Base.load_frames(None, lambda tag: (tmp_path / "nope.nii.gz", tmp_path / "nope_gt.nii.gz", None),
                           {0: 0})
    assert "ED" not in out and "ES" not in out


# --- identify_lv_cavity: geometric pick of the label most enclosed by the myo ring ---
def _ring_with_cavities():
    """A 12x12 mask: a myo ring (LV_MYO=2) around a central cavity (3), plus a separate RV blob (1)
    that does NOT touch the ring. Geometry -> 3 is the LV cavity, 1 is not."""
    m = np.zeros((12, 12), np.uint8)
    m[3:9, 3:9] = LV_MYO           # solid myo block
    m[4:8, 4:8] = 3                # cavity carved inside -> ring of myo around it
    m[0:2, 10:12] = 1             # detached RV blob in the corner
    return m


def test_identify_lv_cavity_picks_enclosed_label():
    """Match class: the label whose shell is most surrounded by myo wins (3 here, not the detached 1)."""
    lv, scores = Base.identify_lv_cavity(_ring_with_cavities())
    assert lv == 3
    assert scores[3] > scores[1]          # cavity shell touches myo; RV blob shell does not
    assert scores[1] == 0.0               # detached blob: no myo contact


def test_identify_lv_cavity_no_foreground_is_none():
    """Empty class: nothing but background (or only myo) -> no candidate label -> None, {}."""
    lv, scores = Base.identify_lv_cavity(np.zeros((4, 4), np.uint8))
    assert lv is None and scores == {}
