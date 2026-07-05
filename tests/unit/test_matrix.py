"""Generalization-matrix runner — locks the two decisions that aren't trivial: the OOD-vs-leak flag
(a cell is OOD iff no manifest subject was in the model's train) and seg_lv class restriction (myo+cav
only, no RV). Model load + validate are stubbed; split_from_cfg and resolve_paths run for real.
"""
import types

import polars as pl
import pytest

from cardioseg.evaluation import matrix
from core.data.static.store import DataCfg


def _meta():
    rows = [("acdc", "s1", "Siemens"), ("acdc", "s2", "Siemens"),
            ("mnms1", "c1", "Canon"), ("mnms1", "c2", "Canon")]
    return pl.DataFrame(
        [{"dataset": d, "subject_id": s, "vendor": v, "labelled": True, "path": f"/s/{d}/{s}.npz"}
         for d, s, v in rows])


@pytest.fixture
def _stub(monkeypatch):
    # a model whose DataCfg holds out Canon -> train = Siemens (s1,s2)
    cfg = types.SimpleNamespace(seed=0, generator=types.SimpleNamespace(
        data=DataCfg(test_vendors=("Canon",), test_manifests=(), test_datasets=(), val_datasets=())))
    monkeypatch.setattr("core.registry.resolve", lambda ref: ref)
    monkeypatch.setattr("core.model.load_run", lambda run, device=None: (object(), cfg, "cpu"))
    monkeypatch.setattr("core.data.static.store.load", lambda *a, **k: _meta())
    # validate: fixed per-class Dice, no EF rows
    monkeypatch.setattr("cardioseg.evaluation.validate.validate",
                        lambda model, paths, size, device, tta=True: ({1: 0.9, 2: 0.8, 3: 0.85}, [], {}))
    manifests = {
        "vendor_canon": {"task": "seg4", "subjects": [["mnms1", "c1"], ["mnms1", "c2"]]},
        "vendor_siemens": {"task": "seg4", "subjects": [["acdc", "s1"], ["acdc", "s2"]]},
        "scd_lv": {"task": "seg_lv", "subjects": [["acdc", "s1"]]},   # reuse a resolvable id
    }
    monkeypatch.setattr("core.data.static.manifest.load", lambda n: manifests[n])
    monkeypatch.setattr("core.data.static.manifest.list_manifests", lambda: list(manifests))
    return None


def test_ood_when_no_manifest_subject_in_train(_stub):
    [r] = matrix.score_matrix(["m"], ["vendor_canon"])
    assert r["ood"] is True and r["n"] == 2                 # Canon held out -> honest OOD
    assert r["dice_mean"] == pytest.approx((0.9 + 0.8 + 0.85) / 3, abs=1e-4)


def test_leak_when_manifest_subject_in_train(_stub):
    [r] = matrix.score_matrix(["m"], ["vendor_siemens"])
    assert r["ood"] is False                                # Siemens WAS trained on -> flagged leak


def test_seg_lv_reports_myo_and_cav_only(_stub):
    [r] = matrix.score_matrix(["m"], ["scd_lv"])
    assert "dice_1" not in r                                # RV dropped for seg_lv
    assert r["dice_2"] == 0.8 and r["dice_3"] == 0.85
    assert r["dice_mean"] == pytest.approx((0.8 + 0.85) / 2, abs=1e-4)
