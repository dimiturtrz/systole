"""Split registry — discover split families, load by name.

    from core.data.ingest.splits import load_split
    from core.data.ingest.split import resolve
    fam = load_split("static_main")
    r = resolve(fam, cloud)                 # -> Resolution(train, val, test, version, test_hash)

A family = a class with `name` + `versions` (see core.data.ingest.split.Split). Add one by importing it here.
"""
from __future__ import annotations

from core.data.ingest.split import resolve
from core.data.ingest.splits.static_all import StaticAll
from core.data.ingest.splits.static_main import StaticMain
from core.data.ingest.splits.synth_main import SynthMain

_FAMILIES = {c.name: c for c in (StaticMain, StaticAll, SynthMain)}


def load_split(name: str):
    if name not in _FAMILIES:
        raise KeyError(f"unknown split {name!r}; have {sorted(_FAMILIES)}")
    return _FAMILIES[name]()


def list_splits() -> list[str]:
    return sorted(_FAMILIES)


def parse_ref(ref: str) -> tuple[str, str | None]:
    """'name@version' -> (name, version); 'name' -> (name, None)."""
    name, ver = (ref.split("@", 1) + [None])[:2]
    return name, ver


def resolve_cfg(d, meta):
    """Resolve a DataCfg's coded split (`d.split`) over `meta` -> Resolution(train, val, test, …).
    The one place the name@version parse + load_split + resolve dance lives."""
    name, ver = parse_ref(d.split)
    return resolve(load_split(name), meta, ver)
