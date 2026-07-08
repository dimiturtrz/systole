"""Harmonize the three datasets' pathology vocabularies to one coarse scheme.

ACDC / M&M-2 / M&Ms-1 each label disease differently (only NOR + HCM match verbatim). To
stratify pathology *across* datasets we collapse to a small clinically-sensible set. This is a
coarse judgment call — edit freely; unknown/ambiguous codes fall to 'other' (a valid value).

Raw vocabularies seen:
  acdc : NOR DCM HCM MINF RV(=ARV)
  mnm2 : NOR LV HCM ARR FALL CIA RV TRI
  mnms1: NOR DCM HCM ARV HHD IHD AHS LVNC Other
"""

# raw code (upper-cased) -> coarse class
_MAP = {
    "NOR": "normal",
    "DCM": "dilated", "LV": "dilated",                 # LV (M&M-2) = LV/dilated dysfunction
    "HCM": "hypertrophic", "HHD": "hypertrophic",      # HHD = hypertensive -> hypertrophic
    "MINF": "ischemic", "IHD": "ischemic",             # infarct / ischemic
    "RV": "rv_congenital", "ARV": "rv_congenital", "ARR": "rv_congenital",
    "FALL": "rv_congenital", "TRI": "rv_congenital",   # Fallot / tricuspid / arrhythmogenic-RV
    # ambiguous -> other: CIA, AHS, LVNC, Other
}


def harmonize(raw: str | None) -> str:
    """Map a raw pathology code to the coarse scheme. None/unknown -> 'other'."""
    if raw is None:
        return "other"
    return _MAP.get(str(raw).strip().upper(), "other")
