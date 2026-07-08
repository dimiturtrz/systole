"""Pure cardioview IO-logic: the square-grid fit, patient-dir resolution, and the mask-assembly
seg pipeline (source-branching + largest-CC + {ED,ES} dict). load_model / model_dir / log_setup are
registry/GPU/logging shells (pragma'd); `masks` runs on a stubbed predict_volume so it's CPU-only."""
import common as C
import numpy as np
import pytest
from common import SIZE, masks, patient_dir, square_stack

# --- square_stack ----------------------------------------------------------

def test_square_stack_pads_small_slices_to_grid():
    """Needs-fit class: slices smaller than SIZE are centre-padded to [D,SIZE,SIZE]; content kept."""
    vol = np.ones((3, 10, 10), np.float32)
    out = square_stack(vol)
    assert out.shape == (3, SIZE, SIZE)
    assert out.sum() == 3 * 10 * 10  # every original voxel survives (padded with 0)


def test_square_stack_crops_oversize_slices():
    """Already-bigger class: slices larger than SIZE are centre-cropped down to SIZE."""
    vol = np.ones((2, SIZE + 40, SIZE + 40), np.float32)
    out = square_stack(vol)
    assert out.shape == (2, SIZE, SIZE)


def test_square_stack_casts_dtype():
    """dtype class: an explicit dtype casts the output (masks want uint8, not float)."""
    vol = np.ones((1, 4, 4), np.float32)
    assert square_stack(vol, np.uint8).dtype == np.uint8
    assert square_stack(vol).dtype == np.float32  # default float32 path


# --- patient_dir -----------------------------------------------------------

def test_patient_dir_accepts_a_full_path(tmp_path):
    """Full-path class: an existing directory is returned as-is (canned list may hold real paths)."""
    d = tmp_path / "patientXX"
    d.mkdir()
    assert patient_dir(str(d)) == d


def test_patient_dir_resolves_bare_id_under_root(tmp_path):
    """Bare-id class: a non-path id is resolved under <root>/acdc/{training,testing}."""
    (tmp_path / "acdc" / "testing" / "patient099").mkdir(parents=True)
    assert patient_dir("patient099", root=str(tmp_path)) == tmp_path / "acdc" / "testing" / "patient099"


def test_patient_dir_missing_raises(tmp_path):
    """Absent class: neither a dir nor found under the root -> FileNotFoundError (not a silent None)."""
    with pytest.raises(FileNotFoundError):
        patient_dir("nope", root=str(tmp_path))


# --- masks (the seg pipeline: source-branch + largest_cc + {ED,ES} dict) ----

def _case(with_ed=True, with_es=True):
    """A consolidated-case dict as preprocess_case returns: ED/ES img (z-scored) + gt label maps."""
    img = np.zeros((2, 8, 8), np.float32)
    gt = np.zeros((2, 8, 8), np.uint8)
    gt[:, 2:5, 2:5] = 3
    c = {}
    if with_ed:
        c["ed_img"], c["ed_gt"] = img, gt
    if with_es:
        c["es_img"], c["es_gt"] = img.copy(), gt.copy()
    return c


def test_masks_gt_source_uses_ground_truth(monkeypatch):
    """GT-source class: source='gt' returns square-stacked GT (never calls the model) for ED and ES."""
    called = []
    monkeypatch.setattr(C, "predict_volume", lambda *a, **k: called.append(1) or None)
    out = masks(_case(), "gt")
    assert set(out) == {"ED", "ES"} and not called
    assert out["ED"].dtype == np.uint8 and out["ED"].max() == 3


def test_masks_pred_source_calls_model_then_largest_cc(monkeypatch):
    """Pred-source class: source='pred' runs predict_volume then largest_cc_per_class per phase.
    Stub predict to a 2-island mask; largest-CC must drop the stray island (EF-biasing clean-up)."""
    pred = np.zeros((2, 8, 8), np.uint8)
    pred[:, 1:4, 1:4] = 3     # big blob
    pred[0, 7, 7] = 3         # stray island (dropped by largest-CC)
    monkeypatch.setattr(C, "predict_volume", lambda *a, **k: pred.copy())
    out = masks(_case(), "pred", model=object(), device="cpu")
    assert set(out) == {"ED", "ES"}
    assert out["ED"][0, 7, 7] == 0        # stray island removed
    assert (out["ED"] == 3).sum() > 0     # main structure kept


def test_masks_skips_absent_phase(monkeypatch):
    """Missing-phase class: an ES-only case -> only 'ES' in the dict (the `continue` on absent img)."""
    monkeypatch.setattr(C, "predict_volume", lambda *a, **k: np.zeros((2, 8, 8), np.uint8))
    out = masks(_case(with_ed=False), "pred", model=object(), device="cpu")
    assert set(out) == {"ES"}
