"""StaticSource + SubjectIds — the static side of the Source seam: content-hash identity
(order-independent, value-sensitive), the subject-key set, path/subject exposure, the raw resident,
and the partial-label validity mask. Mock-fed / I/O-free."""
import polars as pl
import torch

from core.data.ingest.source import StaticSource, SubjectIds

V = pl.col
ids_hash = SubjectIds.ids_hash


def _cloud():
    rows = [("acdc", "s1", "Siemens", True), ("acdc", "s2", "Siemens", True),
            ("mnms1", "c1", "Canon", True), ("mnms1", "g1", "GE", True),
            ("mnms1", "u1", "Canon", False)]                       # unlabelled
    return pl.DataFrame([{"dataset": d, "subject_id": s, "vendor": v, "labelled": lab,
                          "path": f"/s/{d}/{s}.npz"} for d, s, v, lab in rows])


def test_ids_hash_order_independent():
    assert ids_hash([("a", "1"), ("b", "2")]) == ids_hash([("b", "2"), ("a", "1")])
    assert ids_hash([("a", "1")]) != ids_hash([("a", "2")])       # value-sensitive


def test_ids_hash_is_prefixed_sha256():
    h = ids_hash([("acdc", "s1")])
    assert h.startswith("sha256:") and len(h) == len("sha256:") + 64


def test_subject_keys_are_tab_joined_identity_set():
    keys = SubjectIds.subject_keys(_cloud())
    assert "acdc\ts1" in keys and "mnms1\tg1" in keys and len(keys) == 5   # all rows, labelled or not


def test_static_source_exposes_paths_subjects_hash():
    s = StaticSource(_cloud().filter(V("vendor") == "GE"))
    assert s.paths() == ["/s/mnms1/g1.npz"]
    assert s.subjects() == [("mnms1", "g1")] and len(s) == 1
    assert s.provenance()["kind"] == "static" and s.ids_hash().startswith("sha256:")


def test_static_source_ids_hash_matches_subjects():
    s = StaticSource(_cloud().filter(V("labelled")))
    assert s.ids_hash() == ids_hash(s.subjects())                 # the source's hash IS its subjects' hash


def test_static_source_provenance_note_and_count():
    s = StaticSource(_cloud().filter(V("dataset") == "acdc"), "acdc note")
    p = s.provenance()
    assert p["n"] == 2 and p["note"] == "acdc note" and p["ids_hash"] == s.ids_hash()


def test_static_source_resident_is_raw_real(monkeypatch):
    monkeypatch.setattr("core.data.dynamic.dataset.ACDCSliceDataset.load_to_gpu",
                        lambda paths, size, device: (torch.zeros(len(paths), 1, size, size),
                                                     torch.zeros(len(paths), size, size)))
    X, Y = StaticSource(_cloud().filter(V("labelled"))).resident(8, "cpu")
    assert X.shape == (4, 1, 8, 8) and Y.shape == (4, 8, 8)       # raw real, no transforms


def test_static_source_valid_mask_partial():
    cloud = pl.DataFrame([{"dataset": "acdc", "subject_id": "a", "labelled": True, "path": "/p/a.npz"},
                          {"dataset": "scd", "subject_id": "s", "labelled": True, "path": "/p/s.npz"}])
    # 2 slices from acdc (path 0, full), 1 from scd (path 1, LV-only)
    vm = StaticSource(cloud)._valid_mask(torch.tensor([0, 0, 1]), 4, "cpu")
    assert vm.tolist() == [[True, True, True, True], [True, True, True, True], [False, False, True, True]]


def test_static_source_valid_mask_none_when_all_full():
    src = StaticSource(pl.DataFrame([{"dataset": "acdc", "subject_id": "a", "labelled": True, "path": "/p/a.npz"}]))
    assert src._valid_mask(torch.tensor([0, 0]), 4, "cpu") is None   # full-label -> mask-free
