"""Reference-store loader tests (equivalence classes): the graceful-fallback + strict-verified
contract. The point of this store is that missing/unverified knowledge silently falls back to the
per-scan path — so the fallback cases ARE the value."""
from pathlib import Path

from core.data.static.reference import Reference
from core.data.static.reference_build import ReferenceBuild


# --- _range_entry: derived stats leaf (p5/p95/mean/n) + provenance, drops nan/None ---
def test_range_entry_stats_and_provenance():
    e = ReferenceBuild._range_entry([10, 20, 30, 40, 50, None, float("nan")], "cohort X")
    assert e["value"] == [12.0, 48.0]          # p5, p95 over the 5 finite values
    assert e["mean"] == 30.0 and e["n"] == 5   # nan/None dropped
    assert e["extracted_by"] == "computed" and e["verified"] is True
    assert e["based_on"] == "cohort X"


def _write(tmp_path, body: str) -> Path:
    d = tmp_path / "reference"
    d.mkdir()
    (d / "ref.yaml").write_text(body)
    return tmp_path


# --- absent store: everything is the fallback (default) ---
def test_absent_store_falls_back(tmp_path):
    ref = Reference(root=tmp_path / "reference")            # dir doesn't exist
    assert not ref.present()
    assert ref.get("normal_ranges", "ef_normal", default="FB") == "FB"


# --- present + verified: value returned ---
def test_verified_value_used(tmp_path):
    root = _write(tmp_path, "a:\n  b:\n    value: 42\n    verified: true\n")
    ref = Reference(root=root / "reference")
    assert ref.present()
    assert ref.get("a", "b") == 42


# --- present but unverified: STRICT skips it -> fallback ---
def test_unverified_skipped_strict(tmp_path):
    root = _write(tmp_path, "a:\n  b:\n    value: 42\n    verified: false\n")
    assert Reference(root=root / "reference").get("a", "b", default=-1) == -1
    # non-strict caller may opt in to using it
    assert Reference(strict=False, root=root / "reference").get("a", "b") == 42


# --- missing key / non-leaf node -> fallback ---
def test_missing_key_and_nonleaf(tmp_path):
    root = _write(tmp_path, "a:\n  b:\n    value: 42\n    verified: true\n")
    ref = Reference(root=root / "reference")
    assert ref.get("a", "nope", default="FB") == "FB"       # missing key
    assert ref.get("a", default="FB") == "FB"               # 'a' is a branch, not a leaf value


# --- provenance is readable regardless of verified (for audit / model card) ---
def test_provenance_exposed_even_if_unverified(tmp_path):
    root = _write(tmp_path, "a:\n  b:\n    value: 42\n    source: paperX\n    verified: false\n")
    prov = Reference(root=root / "reference").provenance("a", "b")
    assert prov["source"] == "paperX" and prov["verified"] is False
