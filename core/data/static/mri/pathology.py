"""Harmonize the three datasets' pathology vocabularies to one coarse scheme.

ACDC / M&M-2 / M&Ms-1 each label disease differently (only NOR + HCM match verbatim). To
stratify pathology *across* datasets we collapse to a small clinically-sensible set. This is a
coarse judgment call — edit freely; unknown/ambiguous codes fall to 'other' (a valid value).

Raw vocabularies seen:
  acdc : NOR DCM HCM MINF RV(=ARV)
  mnm2 : NOR LV HCM ARR FALL CIA RV TRI
  mnms1: NOR DCM HCM ARV HHD IHD AHS LVNC Other
"""
from enum import StrEnum


class PathologyClass(StrEnum):
    """The coarse cross-dataset pathology scheme (one source of truth for the class strings that were
    scattered across the harmonization map + the stratification/report order)."""
    NORMAL = "normal"
    DILATED = "dilated"
    HYPERTROPHIC = "hypertrophic"
    ISCHEMIC = "ischemic"
    RV_CONGENITAL = "rv_congenital"
    OTHER = "other"


# raw code (upper-cased) -> coarse class
_MAP = {
    "NOR": PathologyClass.NORMAL,
    "DCM": PathologyClass.DILATED, "LV": PathologyClass.DILATED,          # LV (M&M-2) = LV/dilated dysfunction
    "HCM": PathologyClass.HYPERTROPHIC, "HHD": PathologyClass.HYPERTROPHIC,   # HHD = hypertensive -> hypertrophic
    "MINF": PathologyClass.ISCHEMIC, "IHD": PathologyClass.ISCHEMIC,     # infarct / ischemic
    "RV": PathologyClass.RV_CONGENITAL, "ARV": PathologyClass.RV_CONGENITAL, "ARR": PathologyClass.RV_CONGENITAL,
    # Fallot / tricuspid / arrhythmogenic-RV
    "FALL": PathologyClass.RV_CONGENITAL, "TRI": PathologyClass.RV_CONGENITAL,
    # ambiguous -> other: CIA, AHS, LVNC, Other
}


class Pathology:
    """Coarse pathology harmonization across the three datasets' vocabularies (the free helper folded in
    as a staticmethod)."""

    @staticmethod
    def harmonize(raw: str | None) -> PathologyClass:
        """Map a raw pathology code to the coarse scheme. None/unknown -> OTHER."""
        if raw is None:
            return PathologyClass.OTHER
        return _MAP.get(str(raw).strip().upper(), PathologyClass.OTHER)
