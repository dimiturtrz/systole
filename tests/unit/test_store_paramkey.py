"""Processed-cache key: n4 params must enter the key so different N4 settings never collide,
while the no-n4 key stays back-compatible (existing flagship cache resolves)."""
import polars as pl

from core.data import store
from core.data.store import param_key
from core.hparams import N4Cfg


def test_load_coerces_labelled_to_boolean(tmp_path, monkeypatch):
    """Regression (cross-platform): store.load must read `labelled` as Boolean regardless of polars
    schema inference — the newer linux polars read the 'true'/'false' column as String, breaking the
    `pl.col('labelled')` filter. Pinned via schema_overrides; this guards it."""
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path))
    pdir = tmp_path / "processed" / "acdc" / param_key(1.5)
    (pdir / "data").mkdir(parents=True)
    # meta.csv with labelled as text true/false (how it's written) + country for the continent derive
    pl.DataFrame({"subject_id": ["a", "b"], "file": ["a.npz", "b.npz"],
                  "labelled": ["true", "false"], "country": ["Spain", "China"]}).write_csv(pdir / "meta.csv")

    df = store.load(["acdc"], inplane=1.5)
    assert df.schema["labelled"] == pl.Boolean                    # not String
    assert df.filter(pl.col("labelled")).height == 1              # the truthy filter works
    assert df.filter(pl.col("continent") == "Asia").height == 1   # continent derived from country


def test_no_n4_key_unchanged():
    """n4=False -> the original 'inplaneXpY' (no suffix) — existing caches still load."""
    assert param_key(1.5, False) == "inplane1p5"
    assert param_key(1.23, False) == "inplane1p23"


def test_n4_key_encodes_params():
    assert param_key(1.5, True).startswith("inplane1p5_n4")
    assert param_key(1.5, True, N4Cfg()) == "inplane1p5_n4-s4-i50x50x50-f0p15"


def test_n4_distinct_params_distinct_keys():
    """Different N4 settings -> different cache dirs (no stale-cache collision)."""
    base = param_key(1.5, True, N4Cfg())
    assert base != param_key(1.5, True, N4Cfg(shrink=2))
    assert base != param_key(1.5, True, N4Cfg(fwhm=0.3))
    assert base != param_key(1.5, True, N4Cfg(iters=(30, 30, 30)))
