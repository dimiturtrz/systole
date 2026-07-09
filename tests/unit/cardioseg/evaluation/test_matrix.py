"""Generalization-matrix runner — locks the two non-trivial decisions: the OOD-vs-leak flag (a cell
is OOD iff no TestSet subject was in the model's train) and seg_lv class restriction (myo+cav only, no
RV). Model load + validate are stubbed; split_from_cfg and TestSet.source run for real.
"""
import types

import polars as pl
import pytest

from cardioseg.evaluation import matrix
from core.data.ingest.testsets import TestSet
from core.data.static.store import DataCfg

V = pl.col


def _meta():
    rows = [("acdc", "s1", "Siemens"), ("acdc", "s2", "Siemens"),
            ("mnms1", "c1", "Canon"), ("mnms1", "c2", "Canon")]
    return pl.DataFrame(
        [{"dataset": d, "subject_id": s, "vendor": v, "labelled": True, "path": f"/s/{d}/{s}.npz"}
         for d, s, v in rows])


# lock-free TestSets (no drift guard) over the controlled meta
_TS = {
    "canon": TestSet("canon", "seg4", V("vendor") == "Canon"),
    "siemens": TestSet("siemens", "seg4", V("vendor") == "Siemens"),
    "scd_lv": TestSet("scd_lv", "seg_lv", V("dataset") == "acdc"),   # acdc stands in as a resolvable seg_lv
}


@pytest.fixture
def _stub(monkeypatch):
    # a model whose DataCfg holds out Canon -> train = Siemens (s1, s2)
    cfg = types.SimpleNamespace(seed=0, generator=types.SimpleNamespace(
        data=DataCfg(test_vendors=("Canon",), test_datasets=(), val_datasets=())))
    monkeypatch.setattr("core.registry.Registry.resolve", lambda ref: ref)
    monkeypatch.setattr("core.run.Run.load_run", lambda run, device=None: (object(), cfg, "cpu"))
    monkeypatch.setattr("core.data.static.store.build.load", lambda *a, **k: _meta())
    monkeypatch.setattr("cardioseg.evaluation.validate.Evaluator.validate",
                        lambda self, npz_paths: ({1: 0.9, 2: 0.8, 3: 0.85}, [], {}))
    monkeypatch.setattr("cardioseg.evaluation.matrix.TESTSETS", _TS)


def test_ood_when_no_testset_subject_in_train(_stub):
    [r] = matrix.score_matrix(["m"], ["canon"])
    assert r["ood"] is True and r["n"] == 2                 # Canon held out -> honest OOD
    assert r["dice_mean"] == pytest.approx((0.9 + 0.8 + 0.85) / 3, abs=1e-4)


def test_leak_when_testset_subject_in_train(_stub):
    [r] = matrix.score_matrix(["m"], ["siemens"])
    assert r["ood"] is False                                # Siemens WAS trained on -> flagged leak


def test_seg_lv_reports_myo_and_cav_only(_stub):
    [r] = matrix.score_matrix(["m"], ["scd_lv"])
    assert "dice_1" not in r                                # RV dropped for seg_lv
    assert r["dice_2"] == 0.8 and r["dice_3"] == 0.85
    assert r["dice_mean"] == pytest.approx((0.8 + 0.85) / 2, abs=1e-4)
