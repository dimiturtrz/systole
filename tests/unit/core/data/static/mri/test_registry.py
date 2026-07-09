"""Adapter registry (core.data.static.mri.registry, class AdapterRegistry.get_adapter) — the name ->
adapter lookup (equivalence classes: each registered name resolves to its adapter; an unknown name
raises KeyError listing what's available)."""
import pytest

from core.data.static.mri.acdc import AcdcAdapter
from core.data.static.mri.mnm2 import Mnm2Adapter
from core.data.static.mri.registry import AdapterRegistry


# --- known names resolve to the right adapter type, with matching .name ---
def test_get_adapter_known_names():
    a = AdapterRegistry.get_adapter("acdc")
    assert isinstance(a, AcdcAdapter) and a.name == "acdc"
    assert isinstance(AdapterRegistry.get_adapter("mnm2"), Mnm2Adapter)
    for name in ("acdc", "mnm2", "mnms1", "cmrxmotion", "scd"):
        assert AdapterRegistry.get_adapter(name).name == name   # every registered name round-trips


# --- unknown name -> KeyError naming the available set ---
def test_get_adapter_unknown_raises():
    with pytest.raises(KeyError, match="unknown dataset"):
        AdapterRegistry.get_adapter("does_not_exist")
