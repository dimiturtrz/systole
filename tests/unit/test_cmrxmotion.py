"""CMRxMotion adapter tests (equivalence classes): the dataset-specific bits on top of the
shared base helpers — 4D-singleton squeeze, M&Ms label flip, missing-label (severe-motion)
skip, and the new motion_grade axis (worst of ED/ES, from IQA.csv)."""
import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from core.data.static.mri import cmrxmotion as cm


def _nii(path, arr):
    img = nib.Nifti1Image(arr.astype(np.float32), np.eye(4))
    img.header.set_zooms(tuple([1.0] * arr.ndim))
    nib.save(img, str(path))


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A minimal CMRxMotion tree: 2 acquisitions, IQA grades, one missing-label (severe) frame."""
    data = tmp_path / "data"
    # image: 4D with trailing singleton (real layout); label: 3D, raw M&Ms ints
    img4d = np.ones((2, 2, 3, 1))
    raw_lab = np.array([[[0, 1]], [[2, 3]]])             # raw 1=LVcav,2=myo,3=RV
    for case in ("P001-1", "P001-2"):
        d = data / case
        d.mkdir(parents=True)
        for tag in ("ED", "ES"):
            _nii(d / f"{case}-{tag}.nii.gz", img4d)
            # P001-2 ES = severe motion: image present, NO label
            if not (case == "P001-2" and tag == "ES"):
                _nii(d / f"{case}-{tag}-label.nii.gz", raw_lab)
    (tmp_path / "IQA.csv").write_text(
        "Image,Label\nP001-1-ED,1\nP001-1-ES,2\nP001-2-ED,3\nP001-2-ES,3\n")
    monkeypatch.setenv("CARDIAC_CMRX_ROOT", str(tmp_path))
    # csv cache is keyed by path; tmp path is unique per test, so no stale cache.
    return tmp_path


def test_cases_listed_sorted(root):
    cases = cm.cmrx_cases()
    assert [c.name for c in cases] == ["P001-1", "P001-2"]


def test_load_ed_es_squeeze_and_label_flip(root):
    pd = cm.load_ed_es(cm.cmrx_cases()[0])             # P001-1: both frames labelled
    assert "ED" in pd and "ES" in pd
    assert pd["ED"]["img"].ndim == 3                    # 4D singleton -> 3D (frame=0)
    # raw 1<->3 swap reaches canonical (LV-cav 1->3, RV 3->1)
    assert sorted(int(x) for x in np.unique(pd["ED"]["gt"])) == [0, 1, 2, 3]
    assert pd["group"] is None                          # healthy volunteers


def test_missing_label_frame_skipped(root):
    pd = cm.load_ed_es(cm.cmrx_cases()[1])             # P001-2: ES has no label (severe)
    assert "ED" in pd and "ES" not in pd               # severe frame dropped -> store marks unlabelled


def test_motion_grade_worst_of_ed_es(root):
    a = cm.CmrxMotionAdapter()
    cases = {c.name: c for c in cm.cmrx_cases()}
    assert a.meta(cases["P001-1"])["motion_grade"] == "2"   # max(ED=1, ES=2)
    assert a.meta(cases["P001-2"])["motion_grade"] == "3"   # max(3, 3)


def test_meta_fixed_scanner_fields(root):
    m = cm.CmrxMotionAdapter().meta(cm.cmrx_cases()[0])
    assert m["vendor"] == "Siemens" and m["field_T"] == 3.0 and m["scanner"] == "MAGNETOM Vida"
    assert m["country"] == "China" and m["centre"] == "Fudan (Shanghai)"


def test_label_map_is_the_shared_mnm_flip():
    from core.data.static.mri.base import MNM_LABEL_MAP
    assert cm.CmrxMotionAdapter().label_map == MNM_LABEL_MAP == {0: 0, 1: 3, 2: 2, 3: 1}
