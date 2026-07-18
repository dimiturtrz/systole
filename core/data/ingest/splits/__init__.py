"""Split registry — discover split families, load by name.

    from core.data.ingest.splits import Splits
    from core.data.ingest.split import SplitResolver
    fam = Splits.load_split("static_main")
    r = SplitResolver.resolve(fam, cloud)   # -> Resolution(train, val, test, version, test_hash)

A family = a class with `name` + `versions` (see core.data.ingest.split.Split). Add one by importing it here.
"""
from __future__ import annotations

from typing import Any

from core.data.ingest.split import SplitResolver
from core.data.ingest.splits.parametric import Parametric
from core.data.ingest.splits.static_all import StaticAll
from core.data.ingest.splits.static_main import StaticMain
from core.data.ingest.splits.synth_composite import SynthComposite
from core.data.ingest.splits.synth_fov import SynthFov
from core.data.ingest.splits.synth_main import SynthMain

_FAMILIES = {c.name: c for c in (StaticMain, StaticAll, SynthMain, SynthComposite, SynthFov, Parametric)}


class Splits:
    """Split-family registry facade (the free load/list/parse/resolve helpers folded in as
    staticmethods). The `_FAMILIES` registry dict is the module-level source of truth."""

    @staticmethod
    def load_split(name: str):
        if name not in _FAMILIES:
            raise KeyError(f"unknown split {name!r}; have {sorted(_FAMILIES)}")
        return _FAMILIES[name]()

    @staticmethod
    def list_splits() -> list[str]:
        return sorted(_FAMILIES)

    @staticmethod
    def parse_ref(ref: str) -> tuple[str, str | None]:
        """'name@version' -> (name, version); 'name' -> (name, None)."""
        parts = ref.split("@", 1)
        return parts[0], (parts[1] if len(parts) > 1 else None)

    @staticmethod
    def resolve_cfg(d: Any, meta: Any) -> Any:
        """Resolve a DataCfg's coded split (`d.split`) over `meta` -> Resolution(train, val, test, …).
        The one place the name@version parse + load_split + resolve dance lives."""
        name, ver = Splits.parse_ref(d.split)
        return SplitResolver.resolve(Splits.load_split(name), meta, ver)
