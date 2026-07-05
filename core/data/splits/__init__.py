"""Split registry — discover split families, load by name.

    from core.data.splits import load_split
    from core.data.split import resolve
    fam = load_split("static_main")
    r = resolve(fam, cloud)                 # -> Resolution(train, val, test, version, test_hash)

A family = a class with `name` + `versions` (see core.data.split.Split). Add one by importing it here.
"""
from __future__ import annotations

from core.data.splits.static_main import StaticMain
from core.data.splits.synth_main import SynthMain

_FAMILIES = {c.name: c for c in (StaticMain, SynthMain)}


def load_split(name: str):
    if name not in _FAMILIES:
        raise KeyError(f"unknown split {name!r}; have {sorted(_FAMILIES)}")
    return _FAMILIES[name]()


def list_splits() -> list[str]:
    return sorted(_FAMILIES)
