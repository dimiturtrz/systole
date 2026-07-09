"""Pathology harmonization (core.data.static.mri.pathology, class Pathology.harmonize) — the pure
raw-code -> coarse-class map across the three datasets' disease vocabularies (equivalence classes:
known code, alias, unknown passthrough, None, case/whitespace normalization)."""
from core.data.static.mri.pathology import Pathology


# --- known codes -> their coarse class ---
def test_known_codes_map_to_coarse_class():
    assert Pathology.harmonize("NOR") == "normal"
    assert Pathology.harmonize("HCM") == "hypertrophic"
    assert Pathology.harmonize("DCM") == "dilated"
    assert Pathology.harmonize("MINF") == "ischemic"
    assert Pathology.harmonize("RV") == "rv_congenital"


# --- aliases collapse onto the same class ---
def test_aliases_collapse():
    """Cross-dataset synonyms land on one coarse class (LV->dilated, HHD->hypertrophic, IHD->ischemic,
    ARV/ARR/FALL/TRI->rv_congenital)."""
    assert Pathology.harmonize("LV") == "dilated"
    assert Pathology.harmonize("HHD") == "hypertrophic"
    assert Pathology.harmonize("IHD") == "ischemic"
    for code in ("ARV", "ARR", "FALL", "TRI"):
        assert Pathology.harmonize(code) == "rv_congenital"


# --- unknown / ambiguous codes -> 'other' passthrough ---
def test_unknown_codes_fall_to_other():
    for code in ("CIA", "AHS", "LVNC", "Other", "ZZZ"):
        assert Pathology.harmonize(code) == "other"


# --- None -> 'other' (a valid value, never a crash) ---
def test_none_is_other():
    assert Pathology.harmonize(None) == "other"


# --- case + whitespace normalized before lookup ---
def test_case_and_whitespace_normalized():
    assert Pathology.harmonize("  nor  ") == "normal"
    assert Pathology.harmonize("hcm") == "hypertrophic"
