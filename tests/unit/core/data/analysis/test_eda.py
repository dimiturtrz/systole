"""ACDC reality-check + first data viz (core.data.analysis.eda). This is a plotting/reporting CLI
(coverage-omitted): its methods (summarize_patient, save_viz) read real ACDC data / render matplotlib
figures, so there's no pure helper to exercise without I/O. THIN mirror — assert the class + its
staticmethods exist and are callable so the module imports cleanly and its surface stays intact."""
from core.data.analysis.eda import Eda


def test_eda_surface():
    """The reporting/viz staticmethods exist on the class (structural mirror; I/O paths are smoke-tested
    at run time, not here)."""
    for name in ("summarize_patient", "save_viz"):
        assert callable(getattr(Eda, name))
