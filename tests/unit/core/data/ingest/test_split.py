"""SplitResolver — resolves a split family@version into (train, val, test) Sources: highest-semver
_latest, the labelled-complement train, and the full resolve triple (no test/val leak into train)."""
import polars as pl

from core.data.ingest.source import StaticSource
from core.data.ingest.split import Resolution, SplitDef, SplitResolver

V = pl.col


def _cloud():
    rows = [("acdc", "s1", "Siemens", True), ("acdc", "s2", "Siemens", True),
            ("mnms1", "c1", "Canon", True), ("mnms1", "g1", "GE", True),
            ("mnms1", "u1", "Canon", False)]                       # unlabelled -> never selected
    return pl.DataFrame([{"dataset": d, "subject_id": s, "vendor": v, "labelled": lab,
                          "path": f"/s/{d}/{s}.npz"} for d, s, v, lab in rows])


class _Fam:
    name = "fam"
    versions = {"1.0.0": SplitDef(
        test=lambda c: StaticSource(c.filter(V("labelled") & (V("vendor") == "Canon")), "canon"),
        val=lambda c: StaticSource(c.filter(V("labelled") & (V("dataset") == "acdc")), "acdc"))}


def test_latest_picks_highest_semver():
    assert SplitResolver._latest({"1.0.0": None, "1.10.0": None, "1.2.0": None}) == "1.10.0"


def test_complement_is_labelled_rest():
    c = _cloud()
    test = _Fam.versions["1.0.0"].test(c)      # c1
    val = _Fam.versions["1.0.0"].val(c)        # s1, s2
    train = SplitResolver._complement(c, [test, val])
    assert set(train.subjects()) == {("mnms1", "g1")}          # labelled rest; u1 (unlabelled) excluded


def test_resolve_builds_triple_and_complement_train():
    r = SplitResolver.resolve(_Fam(), _cloud())
    assert isinstance(r, Resolution)
    assert set(r.test.subjects()) == {("mnms1", "c1")}
    assert set(r.val.subjects()) == {("acdc", "s1"), ("acdc", "s2")}
    assert set(r.train.subjects()) == {("mnms1", "g1")}        # no test/val leak
    assert r.version == "1.0.0" and r.test_hash == r.test.ids_hash()


def test_resolve_explicit_train_overrides_complement():
    """train=None -> complement; an explicit train callable is used verbatim (the synth arm's dynamic
    train, not the labelled rest)."""
    fam = _Fam()
    fam.versions = dict(_Fam.versions)
    fam.versions["1.0.0"] = SplitDef(
        test=_Fam.versions["1.0.0"].test, val=_Fam.versions["1.0.0"].val,
        train=lambda c: StaticSource(c.filter(V("dataset") == "acdc"), "explicit"))
    r = SplitResolver.resolve(fam, _cloud())
    assert set(r.train.subjects()) == {("acdc", "s1"), ("acdc", "s2")}   # the explicit filter, not complement


def test_resolve_honours_explicit_version():
    fam = _Fam()
    fam.versions = {"1.0.0": _Fam.versions["1.0.0"],
                    "2.0.0": SplitDef(test=_Fam.versions["1.0.0"].val,      # test = acdc in v2
                                      val=_Fam.versions["1.0.0"].test)}
    r = SplitResolver.resolve(fam, _cloud(), version="1.0.0")
    assert r.version == "1.0.0" and set(r.test.subjects()) == {("mnms1", "c1")}
