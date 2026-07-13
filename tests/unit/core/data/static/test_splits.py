"""make_split: test/val held-out-by-criteria vs random val fraction. Plus the LEGACY criteria
split_from_cfg (kept for old-model train reconstruction). Coded splits are tested in test_source_split."""
import types

import polars as pl

from core.data.static.splits import ModelSplit, Splits


def _meta():
    rows = []
    for ds, vendor, n in [("mnm2", "Siemens", 6), ("mnm2", "Philips", 4),
                          ("mnms1", "GE", 3), ("mnms1", "Canon", 2), ("acdc", "Siemens", 5)]:
        rows += [{"subject_id": f"{vendor}{i}", "dataset": ds, "vendor": vendor, "labelled": True}
                 for i in range(n)]
    return pl.DataFrame(rows)


def _cfg(**kw):
    """A DataCfg-like stub carrying only the fields ModelSplit / make_split read."""
    base = {"test_datasets": (), "test_vendors": (), "val_frac": 0.25,
            "val_datasets": (), "val_vendors": (), "train_vendors": (),
            "split": None, "sources": ("mnm2", "mnms1", "acdc")}
    return types.SimpleNamespace(**{**base, **kw})


def test_val_criteria_holds_out_domain():
    """val_datasets/val_vendors -> val is exactly that domain; test its own; train = the rest."""
    tr, val, test = Splits.make_split(_meta(), test_vendors=("Canon", "GE"), val_datasets=("acdc",))
    assert set(test["vendor"].unique()) == {"Canon", "GE"}     # test = unseen vendors
    assert set(val["dataset"].unique()) == {"acdc"}            # val = held-out centre
    assert set(tr["vendor"].unique()) == {"Siemens", "Philips"}  # train = the rest
    assert "acdc" not in tr["dataset"].unique() and "Canon" not in tr["vendor"].unique()


def test_no_val_criteria_falls_back_to_random_frac():
    """Without val criteria, val is a random fraction carved from train (in-domain)."""
    tr, val, test = Splits.make_split(_meta(), test_vendors=("Canon",), val_frac=0.25, seed=0)
    assert len(test) == 2                                       # Canon only
    assert len(val) > 0 and len(tr) > 0
    assert set(val["dataset"].unique()) <= {"mnm2", "mnms1", "acdc"}   # carved from non-test pool


def test_train_vendors_restricts_train_only():
    """train_vendors -> TRAIN keeps only those vendors (scarce/single-vendor regime); val/test intact."""
    full = Splits.make_split(_meta(), test_vendors=("Canon", "GE"), val_datasets=("acdc",))
    scarce = Splits.make_split(_meta(), test_vendors=("Canon", "GE"), val_datasets=("acdc",),
                               train_vendors=("Siemens",))
    assert set(scarce[0]["vendor"].unique()) == {"Siemens"}    # train = Siemens only
    assert len(scarce[0]) < len(full[0])                       # dropped Philips from train
    assert scarce[1].equals(full[1]) and scarce[2].equals(full[2])   # val/test unchanged


def test_modelsplit_split_criteria():
    """ModelSplit.split(): legacy criteria path -> (train, val, test) by vendor/dataset holdout."""
    tr, val, test = ModelSplit(_cfg(test_vendors=("Canon",), val_datasets=("acdc",)), _meta()).split()
    assert set(test["vendor"].unique()) == {"Canon"}            # test = Canon
    assert set(val["dataset"].unique()) == {"acdc"}            # val = acdc
    assert "Canon" not in set(tr["vendor"].unique()) and "acdc" not in set(tr["dataset"].unique())


def test_modelsplit_val_and_test_properties():
    """The .val / .test properties expose the criteria val/test frames a model held out."""
    ms = ModelSplit(_cfg(test_vendors=("Canon",), val_datasets=("acdc",)), _meta())
    assert set(ms.test["vendor"].unique()) == {"Canon"}
    assert set(ms.val["dataset"].unique()) == {"acdc"}


def test_modelsplit_seen_excludes_test_train_excludes_val():
    """seen_keys = train∪val (labelled, in-sources, minus test); train_keys drops val too."""
    ms = ModelSplit(_cfg(test_vendors=("Canon",), val_datasets=("acdc",)), _meta())
    seen, trained = ms.seen_keys(), ms.train_keys()
    assert trained <= seen                                      # train is a subset of seen (val carved off)
    assert not any(k.startswith("mnms1\tCanon") for k in seen)  # test subjects never seen
    assert not any(k.startswith("acdc\t") for k in trained)     # val (acdc) not in train
