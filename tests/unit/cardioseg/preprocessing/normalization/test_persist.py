"""Dataset-metadata persistence (cardioseg.preprocessing.normalization.persist). The adapter +
data_root are faked so the pure provenance/serialize logic runs without real data or sidecars."""
import argparse
from pathlib import Path

from omegaconf import OmegaConf

import cardioseg.preprocessing.normalization.persist as P
from cardioseg.preprocessing.normalization.persist import Persist
from core.data.static.mri.registry import AdapterRegistry


def test_prov_paper_layer_wins():
    """_prov class: a paper-overlay dict for the field -> cited source, by='paper', verified flag honored."""
    p = Persist._prov("field_scanner", {"field_scanner": "auto-src"},
                      {"field_scanner": {"source": "Bernard 2018", "verified": True}})
    assert p == {"source": "Bernard 2018", "by": "paper", "verified": True}


def test_prov_paper_unverified_flag():
    """_prov boundary: paper overlay without verified -> verified False."""
    p = Persist._prov("f", {}, {"f": {"source": "cite"}})
    assert p["by"] == "paper" and p["verified"] is False


def test_prov_auto_source_map():
    """_prov class: no paper entry -> adapter _source map, by='auto', verified True."""
    p = Persist._prov("te", {"te": "dicom"}, {})
    assert p == {"source": "dicom", "by": "auto", "verified": True}


def test_prov_auto_rest_fallback():
    """_prov boundary: field absent from _source -> 'rest'/'all' fallback then 'unknown'."""
    assert Persist._prov("x", {"rest": "sidecar"}, {})["source"] == "sidecar"
    assert Persist._prov("x", {}, {})["source"] == "unknown"


class _Case:
    def __init__(self, name): self.name = name


class _FakeAdapter:
    """Minimal adapter: two cases, each with a meta dict + a _source provenance map."""
    def cases(self):
        return [_Case("p001"), _Case("p002")]

    def meta(self, case):
        return {"scanner": "Siemens", "field_strength": 1.5,
                "_source": {"scanner": "dicom", "rest": "parsed"}}


def test_persist_writes_provenance_yaml(monkeypatch, tmp_path):
    """persist_meta pipeline: fake adapter -> yaml written under raw/, provenance-tagged as expected."""
    monkeypatch.setattr(AdapterRegistry, "get_adapter", lambda _ds: _FakeAdapter())
    monkeypatch.setattr(P.Config, "data_root", staticmethod(lambda _kind="raw": str(tmp_path)))
    monkeypatch.setattr(P.Persist, "_overlay",
                        lambda: {"acdc": {"scanner": {"source": "Bernard 2018", "verified": True}}})

    out = Persist.persist_meta("acdc")
    assert out == Path(tmp_path) / "acdc" / "meta" / "acdc.yaml"
    assert out.exists()

    loaded = OmegaConf.to_container(OmegaConf.load(out))
    assert loaded["dataset"] == "acdc" and loaded["n"] == 2
    subj = loaded["subjects"]["p001"]
    assert subj["scanner"]["value"] == "Siemens"
    assert subj["scanner"]["by"] == "paper"            # overlay won for scanner
    assert subj["field_strength"]["by"] == "auto"      # no overlay -> auto
    assert "_source" not in subj                       # provenance map popped, not serialized


def test_overlay_absent_is_empty(monkeypatch, tmp_path):
    """_overlay boundary: missing sources.yaml -> empty dict."""
    monkeypatch.setattr(P, "_SOURCES", tmp_path / "missing.yaml")
    assert P.Persist._overlay() == {}


def test_run_single_dataset(monkeypatch):
    """run: --dataset X -> persist_meta called for that one dataset."""
    monkeypatch.setattr(P, "_DATASETS", ("acdc", "mnm2"))
    called = []
    monkeypatch.setattr(P.Persist, "persist_meta", lambda ds: called.append(ds))
    P.Persist.run(argparse.Namespace(dataset="acdc"))
    assert called == ["acdc"]


def test_run_all_datasets(monkeypatch):
    """run: default 'all' -> persist_meta over every known dataset."""
    monkeypatch.setattr(P, "_DATASETS", ("acdc", "mnm2"))
    called = []
    monkeypatch.setattr(P.Persist, "persist_meta", lambda ds: called.append(ds))
    P.Persist.run(argparse.Namespace(dataset="all"))
    assert called == ["acdc", "mnm2"]
