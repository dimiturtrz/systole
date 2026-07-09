"""Consolidated store pure logic (core.data.static.store) — equivalence classes over the meta-schema
derivations (BSA, age-band, vendor-norm, region, labelled), the acquisition-mine grouping/gate, and
the meta-row assembly. All I/O-free: fed synthetic dicts + in-memory polars frames (the fake-cloud
idiom from test_source_split). The npz/glob shells (build/load/_write_meta reads) are integration."""
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from core.data.static.mri.registry import AdapterRegistry
from core.data.static.store.build import Build
from core.data.static.store.normalize import Normalizer
from core.data.static.store.query import (
    AcqReference,
    MetaBuilder,
    Store,
)

_region_of = MetaBuilder._region_of
_bsa = MetaBuilder._bsa
_age_band = MetaBuilder._age_band
_norm_vendor = MetaBuilder._norm_vendor
_is_labelled = MetaBuilder._is_labelled
param_key = Store.param_key
dataset_dir = Store.dataset_dir
load_arrays = Store.load_arrays
load = Build.load


# --- _region_of: mapped country / unmapped / null ---
def test_region_of_classes():
    assert _region_of("France") == "Europe"          # mapped Europe
    assert _region_of("Japan") == "Asia"             # mapped Asia
    assert _region_of("USA") == "North America"      # mapped America
    assert _region_of("Brazil") is None              # unmapped country
    assert _region_of(None) is None                  # null country -> null region
    assert _region_of("") is None                    # empty falsy -> null


# --- _bsa: valid / one-missing / non-numeric / non-positive ---
def test_bsa_valid_mosteller():
    assert _bsa(180, 80) == pytest.approx((180 * 80 / 3600) ** 0.5, abs=0.01)   # ~2.0


def test_bsa_missing_or_bad():
    assert _bsa(None, 80) is None                     # height missing
    assert _bsa(180, None) is None                    # weight missing
    assert _bsa("tall", 80) is None                   # non-numeric
    assert _bsa(0, 80) is None                         # non-positive height -> None
    assert _bsa(180, -5) is None                       # non-positive weight -> None


# --- _age_band: each bucket + both boundaries + null ---
def test_age_band_buckets_and_boundaries():
    assert _age_band(30) == "<45"
    assert _age_band(44.9) == "<45"
    assert _age_band(45) == "45-60"                    # lower boundary inclusive
    assert _age_band(59) == "45-60"
    assert _age_band(60) == "60-75"                    # boundary
    assert _age_band(74) == "60-75"
    assert _age_band(75) == "75+"                      # boundary
    assert _age_band(90) == "75+"
    assert _age_band(None) is None                     # missing
    assert _age_band("old") is None                    # garbage


# --- _norm_vendor: each known vendor (substring, case) / passthrough / null ---
def test_norm_vendor_known_and_case():
    assert _norm_vendor("SIEMENS Healthineers") == "Siemens"   # substring + case-fold
    assert _norm_vendor("philips medical") == "Philips"
    assert _norm_vendor("GE MEDICAL SYSTEMS") == "GE"
    assert _norm_vendor("Canon Inc") == "Canon"


def test_norm_vendor_unknown_and_null():
    assert _norm_vendor("Hitachi") == "Hitachi"        # unknown -> passthrough
    assert _norm_vendor(None) is None                  # null
    assert _norm_vendor("") is None                    # empty falsy -> None


# --- _is_labelled: both present / one empty / one missing ---
def _gt(has):
    return (np.array([[0, 2, 3]]) if has else np.zeros((1, 3), int))


def test_is_labelled_both_nonempty():
    assert _is_labelled({"ed_gt": _gt(True), "es_gt": _gt(True)}) is True


def test_is_labelled_es_empty():
    assert _is_labelled({"ed_gt": _gt(True), "es_gt": _gt(False)}) is False   # ES all-zero


def test_is_labelled_missing_key():
    assert _is_labelled({"ed_gt": _gt(True)}) is False   # es_gt absent -> None -> not labelled


# --- _meta_row: derivations wired through (vendor-norm, region, bsa, age-band, labelled) ---
def test_meta_row_derivations():
    case = Path("/x/subjX")
    meta = {"vendor": "SIEMENS", "group": "DCM", "country": "France", "age": 50,
            "height": 180, "weight": 80, "field_T": 1.5}
    arrays = {"ed_gt": np.array([[2, 3]]), "es_gt": np.array([[2, 3]])}
    r = MetaBuilder("acdc", None)._row(case, arrays, meta, "subjX.npz")
    assert r["subject_id"] == "subjX" and r["dataset"] == "acdc" and r["raw_path"] == str(case)
    assert r["vendor"] == "Siemens"                    # normalized
    assert r["region"] == "Europe" and r["age_band"] == "45-60"
    assert r["bsa"] == pytest.approx(2.0, abs=0.05) and r["labelled"] is True
    assert r["pathology_raw"] == "DCM"                 # raw group preserved


def test_meta_row_field_list_joined():
    """field_T as a list (multi-field dataset) -> slash-joined string, not a raw list."""
    r = MetaBuilder("mnm2", None)._row(Path("/x/s"), {"ed_gt": np.zeros((1, 2), int)},
                  {"field_T": [1.5, 3.0]}, "s.npz")
    assert r["field_T"] == "1.5/3.0" and r["labelled"] is False   # empty gt -> unlabelled


# --- acquisition_from_frame: bSSFP gate, null skip, field folding, no-column ---
def _acq_frame(rows):
    return pl.DataFrame(rows, schema={"vendor": pl.Utf8, "field_T": pl.Utf8, "tr_ms": pl.Utf8,
                                      "te_ms": pl.Utf8, "flip_deg": pl.Utf8}, strict=False)


def test_acquisition_no_tr_column():
    assert AcqReference.from_frame(pl.DataFrame({"vendor": ["GE"]})) == {}   # no tr_ms -> {}


def test_acquisition_gate_rejects_out_of_range_and_nulls():
    """Only in-gate bSSFP TR (2-6ms) with non-null vendor+TR survives; GRE (39ms) + nulls dropped."""
    df = _acq_frame([
        {"vendor": "GE", "field_T": "1.5", "tr_ms": "3.0", "te_ms": "1.5", "flip_deg": "50"},  # keep
        {"vendor": "GE", "field_T": "1.5", "tr_ms": "39.0", "te_ms": "5", "flip_deg": "12"},   # GRE drop
        {"vendor": None, "field_T": "1.5", "tr_ms": "3.1", "te_ms": "1.5", "flip_deg": "50"},  # null vendor
        {"vendor": "GE", "field_T": "1.5", "tr_ms": None, "te_ms": "1.5", "flip_deg": "50"},   # null tr
    ])
    acq = AcqReference.from_frame(df)
    assert set(acq) == {"GE"}
    assert acq["GE"]["tr_ms"]["value"] == 3.0                        # only the kept row's median
    assert acq["GE"]["tr_ms"]["source"] == "DICOM-measured" and acq["GE"]["tr_ms"]["verified"] is True
    assert "1 DICOM subjects" in acq["GE"]["tr_ms"]["based_on"]      # provenance count


def test_acquisition_field_folding_and_flip_bucket():
    """'1.5' and '1.500000' fold into ONE group (round to 1dp); 1.5T -> flip_deg_1p5t, 3T -> _3t."""
    df = _acq_frame([
        {"vendor": "GE", "field_T": "1.5", "tr_ms": "3.0", "te_ms": "1.5", "flip_deg": "50"},
        {"vendor": "GE", "field_T": "1.500000", "tr_ms": "4.0", "te_ms": "1.5", "flip_deg": "60"},
        {"vendor": "Siemens", "field_T": "3.0", "tr_ms": "3.0", "te_ms": "1.5", "flip_deg": "45"},
    ])
    acq = AcqReference.from_frame(df)
    assert acq["GE"]["tr_ms"]["value"] == 3.5                        # median(3,4) — ONE folded group
    assert "flip_deg_1p5t" in acq["GE"] and "flip_deg_3t" not in acq["GE"]
    assert "flip_deg_3t" in acq["Siemens"] and "flip_deg_1p5t" not in acq["Siemens"]


# --- load: labelled-boolean coercion + continent derive (I/O shell over a written meta.csv) ---
def test_load_coerces_labelled_and_derives_continent(tmp_path, monkeypatch):
    """store.load reads labelled as Boolean (not String) and derives continent from country."""
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path))
    pdir = tmp_path / "processed" / "acdc" / param_key(1.5)
    (pdir / "data").mkdir(parents=True)
    pl.DataFrame({"subject_id": ["a", "b"], "file": ["a.npz", "b.npz"],
                  "labelled": ["true", "false"], "country": ["Spain", "China"]}).write_csv(pdir / "meta.csv")
    df = load(["acdc"], inplane=1.5)
    assert df.schema["labelled"] == pl.Boolean
    assert df.filter(pl.col("labelled")).height == 1
    assert df.filter(pl.col("continent") == "Asia").height == 1


def test_load_nyul_without_reference_raises(tmp_path, monkeypatch):
    """nyul=True with no reference/nyul.yaml -> RuntimeError (can't harmonize without the standard)."""
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path))
    with pytest.raises(RuntimeError, match="fit it first"):
        load(["acdc"], nyul=True)


# --- dataset_dir: paramkey folder composition ---
def test_dataset_dir_encodes_paramkey(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path))
    d = dataset_dir("acdc", 1.5, n4=True)
    assert d.name.startswith("inplane1p5_n4") and d.parent.name == "acdc"


# --- load_arrays: npz round-trip, group -> plain scalar ---
def test_load_arrays_group_scalar(tmp_path):
    p = tmp_path / "s.npz"
    np.savez(p, ed_img=np.zeros((2, 2)), group=np.array("DCM"))
    d = load_arrays(p)
    assert d["group"] == "DCM" and not isinstance(d["group"], np.ndarray)   # 0-d -> py scalar
    assert d["ed_img"].shape == (2, 2)


# --- _write_meta: materializes meta.csv over written subjects via a fake adapter ---
class _FakeAdapter:
    def __init__(self, cases):
        self._cases = cases

    def cases(self):
        return self._cases

    def meta(self, case):
        return {"vendor": "SIEMENS", "group": "NOR", "age": 50}


def test_write_meta_over_written_subjects(tmp_path):
    data_dir = tmp_path / "data"; data_dir.mkdir()
    for name in ("s1", "s2"):
        np.savez(data_dir / f"{name}.npz", ed_gt=np.array([[2, 3]]), es_gt=np.array([[2, 3]]))
    case_missing = tmp_path / "s3"                            # no npz -> skipped
    adapter = _FakeAdapter([data_dir.parent / "s1", data_dir.parent / "s2", case_missing])
    out = MetaBuilder("acdc", adapter).write(data_dir, tmp_path)
    df = pl.read_csv(out, schema_overrides={"labelled": pl.Boolean})
    assert df.height == 2                                     # s3 (no npz) skipped
    assert set(df["subject_id"]) == {"s1", "s2"}
    assert df["vendor"].to_list() == ["Siemens", "Siemens"]  # normalized through _meta_row
    assert df.filter(pl.col("labelled")).height == 2         # both gts non-empty


# --- migrate_meta: skips unregistered processed dirs (no adapter) ---
def test_migrate_meta_skips_unregistered(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path))
    pdir = tmp_path / "processed" / "mrxcat_pool" / "inplane1p5"   # not a registered adapter
    (pdir / "data").mkdir(parents=True)
    (pdir / "meta.csv").write_text("subject_id\nx\n")
    assert MetaBuilder.migrate() == []                              # unregistered -> skipped, no crash


def _fake_processed(tmp_path, name):
    data_dir = tmp_path / "processed" / name / "inplane1p5" / "data"
    data_dir.mkdir(parents=True)
    np.savez(data_dir / "s1.npz", ed_gt=np.array([[2, 3]]), es_gt=np.array([[2, 3]]))
    (data_dir.parent / "meta.csv").write_text("subject_id\nold\n")   # stale, pre-migration
    return data_dir.parent


def test_migrate_meta_rewrites_registered(tmp_path, monkeypatch):
    """A registered adapter's processed dir is re-emitted with the current schema (no image reload)."""
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path))
    _fake_processed(tmp_path, "acdc")
    monkeypatch.setattr(AdapterRegistry, "get_adapter", lambda n: _FakeAdapter([tmp_path / "processed" / n / "inplane1p5" / "data" / "s1"]))
    out = MetaBuilder.migrate()
    assert len(out) == 1 and out[0].name == "meta.csv"
    df = pl.read_csv(out[0], schema_overrides={"labelled": pl.Boolean})
    assert df["subject_id"].to_list() == ["s1"] and df["vendor"].to_list() == ["Siemens"]


def test_migrate_meta_names_filter(tmp_path, monkeypatch):
    """`names` restricts which stores refresh; a non-matching processed dir is skipped."""
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path))
    _fake_processed(tmp_path, "acdc")
    monkeypatch.setattr(AdapterRegistry, "get_adapter", lambda n: _FakeAdapter([tmp_path / "processed" / n / "inplane1p5" / "data" / "s1"]))
    assert MetaBuilder.migrate(["mnm2"]) == []                       # acdc present but not in names -> skipped


# --- _nyul_ref_path: reference-dir composition ---
def test_nyul_ref_path(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path))
    assert Normalizer.ref_path().name == "nyul.yaml"        # under the reference dir


# --- _write_meta: adapter.meta() raising -> empty meta, row still emitted (not fatal) ---
class _RaisingMetaAdapter:
    def __init__(self, cases):
        self._cases = cases

    def cases(self):
        return self._cases

    def meta(self, case):
        raise KeyError("missing sidecar field")


def test_write_meta_tolerates_meta_error(tmp_path):
    data_dir = tmp_path / "data"; data_dir.mkdir()
    np.savez(data_dir / "s1.npz", ed_gt=np.array([[2, 3]]), es_gt=np.array([[2, 3]]))
    out = MetaBuilder("acdc", _RaisingMetaAdapter([tmp_path / "s1"])).write(data_dir, tmp_path)
    df = pl.read_csv(out, schema_overrides={"labelled": pl.Boolean})
    assert df.height == 1 and df["subject_id"].to_list() == ["s1"]   # row emitted despite meta() raising
    assert df["vendor"].to_list() == [None]                          # no meta -> null-derived fields
