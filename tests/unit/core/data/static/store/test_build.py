"""Store BUILD + cloud-load engine (core.data.static.store.build) — the load-side equivalence classes:
labelled-Boolean coercion + continent derive over a written meta.csv, the nyul-without-reference guard,
process-if-missing build (skips already-written subjects, re-emits meta.csv), and load_cfg threading ALL
of a DataCfg's preprocessing params. I/O over a tmp_path CARDIAC_DATA + a fake adapter/normalizer — no
real DICOM/NIfTI reads."""
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from core.data.static.mri.registry import AdapterRegistry
from core.data.static.store import build as build_mod
from core.data.static.store.build import Build
from core.data.static.store.normalize import Normalizer
from core.data.static.store.query import DataCfg, Recipe, Store

def param_key(inplane, **recipe):
    return Store(Recipe(inplane=inplane, **recipe)).param_key()


load = Build.load


# --- load: labelled-Boolean coercion + continent derive (I/O shell over a written meta.csv) ---
def test_load_coerces_labelled_and_derives_continent(tmp_path, monkeypatch):
    """store.load reads labelled as Boolean (not String) and derives continent from country.
    Regression (cross-platform): newer linux polars read the 'true'/'false' column as String,
    breaking the `pl.col('labelled')` filter — pinned via schema_overrides."""
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path))
    pdir = tmp_path / "processed" / "acdc" / param_key(1.5)
    (pdir / "data").mkdir(parents=True)
    # meta.csv with labelled as text true/false (how it's written) + country for the continent derive
    pl.DataFrame({"subject_id": ["a", "b"], "file": ["a.npz", "b.npz"],
                  "labelled": ["true", "false"], "country": ["Spain", "China"]}).write_csv(pdir / "meta.csv")
    df = load(["acdc"], Recipe(inplane=1.5))
    assert df.schema["labelled"] == pl.Boolean                    # not String
    assert df.filter(pl.col("labelled")).height == 1              # the truthy filter works
    assert df.filter(pl.col("continent") == "Asia").height == 1   # continent derived from country


def test_load_nyul_without_reference_raises(tmp_path, monkeypatch):
    """nyul=True with no reference/nyul.yaml -> RuntimeError (can't harmonize without the standard)."""
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path))
    with pytest.raises(RuntimeError, match="fit it first"):
        load(["acdc"], Recipe(nyul=True))


# --- build: process-if-missing (skip already-written subjects) + meta.csv (re)emit ---
class _FakeAdapter:
    def __init__(self, cases):
        self._cases = cases

    def cases(self):
        return self._cases

    def load_ed_es(self, case):
        return {}

    def meta(self, case):
        return {"vendor": "SIEMENS", "group": "NOR", "age": 50}


def test_build_process_if_missing(tmp_path, monkeypatch):
    """Only subjects WITHOUT an npz are consolidated (process-if-missing); a pre-written one is not
    re-normalized. meta.csv is (re)emitted over every written subject."""
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path))
    cases = [Path("s1"), Path("s2")]
    monkeypatch.setattr(AdapterRegistry, "get_adapter", lambda n: _FakeAdapter(cases))

    normalized = []

    def _fake_apply(self, case, loader):
        normalized.append(case.name)
        return {"ed_gt": np.array([[2, 3]]), "es_gt": np.array([[2, 3]]), "patient": "drop"}

    monkeypatch.setattr(Normalizer, "apply_case", _fake_apply)

    out = Build.build("acdc", Recipe(inplane=1.5))
    assert sorted(normalized) == ["s1", "s2"]                     # both fresh -> both normalized
    assert (out / "data" / "s1.npz").exists() and (out / "meta.csv").exists()
    # `patient` key dropped from the saved npz
    assert "patient" not in np.load(out / "data" / "s1.npz").files

    normalized.clear()
    out2 = Build.build("acdc", Recipe(inplane=1.5))                              # second call: both present
    assert normalized == []                                       # nothing re-normalized
    assert out2 == out


def test_build_rebuild_forces_all(tmp_path, monkeypatch):
    """rebuild=True re-normalizes every case even when the npz already exists."""
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path))
    cases = [Path("s1")]
    monkeypatch.setattr(AdapterRegistry, "get_adapter", lambda n: _FakeAdapter(cases))
    seen = []
    monkeypatch.setattr(Normalizer, "apply_case",
                        lambda self, case, loader: seen.append(case.name) or
                        {"ed_gt": np.array([[2, 3]]), "es_gt": np.array([[2, 3]])})
    Build.build("acdc", Recipe(inplane=1.5))
    Build.build("acdc", Recipe(inplane=1.5), rebuild=True)
    assert seen == ["s1", "s1"]                                   # normalized twice (once + forced rebuild)


# --- load_cfg: threads ALL of DataCfg's preprocessing params into load ---
def test_load_cfg_threads_all_params(monkeypatch):
    """load_cfg must forward inplane/n4/n4_params/nyul/norm/sources — not silently read zscore npz for a
    nyul/blood-norm model. Captures the load call kwargs."""
    captured = {}

    def _fake_load(names=None, recipe=None, **kw):
        captured.update(names=names, recipe=recipe, **kw)
        return pl.DataFrame({"x": [1]})

    monkeypatch.setattr(build_mod.Build, "load", staticmethod(_fake_load))
    d = DataCfg(sources=("acdc", "mnm2"), inplane=1.25, n4=True, nyul=False, norm="blood")
    Build.load_cfg(d, workers=3)
    assert captured["names"] == ["acdc", "mnm2"]
    assert captured["recipe"].inplane == 1.25 and captured["recipe"].n4 is True
    assert captured["recipe"].n4_params == d.n4_params
    assert captured["recipe"].nyul is False and captured["recipe"].norm == "blood" and captured["workers"] == 3


def test_load_cfg_sources_override(monkeypatch):
    """`sources` arg overrides d.sources (e.g. the matrix's full eval cloud)."""
    captured = {}
    monkeypatch.setattr(build_mod.Build, "load",
                        staticmethod(lambda names=None, recipe=None, **kw: captured.update(names=names) or pl.DataFrame({"x": [1]})))
    d = DataCfg(sources=("acdc",))
    Build.load_cfg(d, sources=("mnm2", "mnms1"))
    assert captured["names"] == ["mnm2", "mnms1"]                 # override, not d.sources
