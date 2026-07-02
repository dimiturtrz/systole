"""make_split: test/val held-out-by-criteria vs random val fraction."""
import polars as pl

from core.data.static.splits import make_split


def _meta():
    rows = []
    for ds, vendor, n in [("mnm2", "Siemens", 6), ("mnm2", "Philips", 4),
                          ("mnms1", "GE", 3), ("mnms1", "Canon", 2), ("acdc", "Siemens", 5)]:
        rows += [{"subject_id": f"{vendor}{i}", "dataset": ds, "vendor": vendor, "labelled": True}
                 for i in range(n)]
    return pl.DataFrame(rows)


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
