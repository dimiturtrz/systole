"""make_split: test/val held-out-by-criteria vs random val fraction. Plus named splits + the
frozen-manifest test path (split_from_cfg)."""
import types

import polars as pl

from core.data.static import splits as S
from core.data.static.splits import make_split, named_split, split_from_cfg


def _meta():
    rows = []
    for ds, vendor, n in [("mnm2", "Siemens", 6), ("mnm2", "Philips", 4),
                          ("mnms1", "GE", 3), ("mnms1", "Canon", 2), ("acdc", "Siemens", 5)]:
        rows += [{"subject_id": f"{vendor}{i}", "dataset": ds, "vendor": vendor, "labelled": True}
                 for i in range(n)]
    return pl.DataFrame(rows)


def _cfg(**kw):
    """A DataCfg-like stub carrying only the fields split_from_cfg reads."""
    base = dict(test_manifests=(), test_datasets=(), test_vendors=(), val_frac=0.25,
                val_datasets=(), val_vendors=(), train_vendors=())
    return types.SimpleNamespace(**{**base, **kw})


def test_val_criteria_holds_out_domain():
    """val_datasets/val_vendors -> val is exactly that domain; test its own; train = the rest."""
    tr, val, test = make_split(_meta(), test_vendors=("Canon", "GE"), val_datasets=("acdc",))
    assert set(test["vendor"].unique()) == {"Canon", "GE"}     # test = unseen vendors
    assert set(val["dataset"].unique()) == {"acdc"}            # val = held-out centre
    assert set(tr["vendor"].unique()) == {"Siemens", "Philips"}  # train = the rest
    assert "acdc" not in tr["dataset"].unique() and "Canon" not in tr["vendor"].unique()


def test_no_val_criteria_falls_back_to_random_frac():
    """Without val criteria, val is a random fraction carved from train (in-domain)."""
    tr, val, test = make_split(_meta(), test_vendors=("Canon",), val_frac=0.25, seed=0)
    assert len(test) == 2                                       # Canon only
    assert len(val) > 0 and len(tr) > 0
    assert set(val["dataset"].unique()) <= {"mnm2", "mnms1", "acdc"}   # carved from non-test pool


def test_train_vendors_restricts_train_only():
    """train_vendors -> TRAIN keeps only those vendors (scarce/single-vendor regime); val/test intact."""
    full = make_split(_meta(), test_vendors=("Canon", "GE"), val_datasets=("acdc",))
    scarce = make_split(_meta(), test_vendors=("Canon", "GE"), val_datasets=("acdc",),
                        train_vendors=("Siemens",))
    assert set(scarce[0]["vendor"].unique()) == {"Siemens"}    # train = Siemens only
    assert len(scarce[0]) < len(full[0])                       # dropped Philips from train
    assert scarce[1].equals(full[1]) and scarce[2].equals(full[2])   # val/test unchanged


def test_named_split_builds_datacfg():
    """A named preset -> a DataCfg with the split's fields (test pointed at frozen manifests)."""
    d = named_split("xvendor")
    assert d.test_manifests == ("vendor_canon", "vendor_ge", "dataset_cmrxmotion")
    assert d.test_datasets == () and d.test_vendors == () and d.val_datasets == ("acdc",)


def test_split_from_cfg_frozen_manifest_test(monkeypatch):
    """test_manifests set -> TEST = the manifest's subjects, carved OUT of train; val by criteria."""
    monkeypatch.setattr("core.data.static.manifest.load",
                        lambda name: {"subjects": [["mnms1", "Canon0"], ["mnms1", "Canon1"]]})
    d = _cfg(test_manifests=("m",), val_datasets=("acdc",))
    tr, val, test, missing = split_from_cfg(d, _meta(), seed=0)
    assert set(test["vendor"].unique()) == {"Canon"} and len(test) == 2
    assert "Canon" not in set(tr["vendor"].unique())           # test subjects not in train (no leak)
    assert set(val["dataset"].unique()) == {"acdc"} and missing == []


def test_split_from_cfg_flags_drift(monkeypatch):
    """A frozen id absent from the store -> reported in `missing`, never silently dropped."""
    monkeypatch.setattr("core.data.static.manifest.load",
                        lambda name: {"subjects": [["mnms1", "Canon0"], ["mnms1", "GHOST9"]]})
    tr, val, test, missing = split_from_cfg(_cfg(test_manifests=("m",)), _meta(), seed=0)
    assert missing == [["mnms1", "GHOST9"]] and len(test) == 1


def test_split_from_cfg_empty_manifests_falls_back_to_criteria():
    """No test_manifests -> criteria make_split (back-compat), missing = []."""
    tr, val, test, missing = split_from_cfg(_cfg(test_vendors=("Canon",)), _meta(), seed=0)
    assert set(test["vendor"].unique()) == {"Canon"} and missing == []
