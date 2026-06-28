"""Processed-cache key: n4 params must enter the key so different N4 settings never collide,
while the no-n4 key stays back-compatible (existing flagship cache resolves)."""
from cardioseg.data.store import param_key
from core.hparams import N4Cfg


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
