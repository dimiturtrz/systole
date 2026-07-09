"""ReferenceBuild staticmethod tests (core.data.static.reference_build) — the derived-stats leaf
builder that computes p5/p95/mean/n over a cohort with provenance, dropping nan/None."""
from core.data.static.reference_build import ReferenceBuild


# --- _range_entry: derived stats leaf (p5/p95/mean/n) + provenance, drops nan/None ---
def test_range_entry_stats_and_provenance():
    e = ReferenceBuild._range_entry([10, 20, 30, 40, 50, None, float("nan")], "cohort X")
    assert e["value"] == [12.0, 48.0]          # p5, p95 over the 5 finite values
    assert e["mean"] == 30.0 and e["n"] == 5   # nan/None dropped
    assert e["extracted_by"] == "computed" and e["verified"] is True
    assert e["based_on"] == "cohort X"
