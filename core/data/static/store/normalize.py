"""The normalization stage — the ONLY store module that imports the preprocessing pipeline (N4/Nyúl/
resample). Conceptually a DOWNSTREAM stage (raw -> processed[geometry] -> normalized -> ...), inverted
out of the read side so `query` stays preprocessing-free: `build` constructs a `Normalizer` from a
recipe and delegates per-case, never importing `core.preprocessing` itself.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from core.config import DEFAULT_INPLANE
from core.data.static.mri.base import Phase
from core.data.static.mri.registry import AdapterRegistry
from core.data.static.reference import Reference
from core.data.static.store.query import SOURCE_DATASETS, Recipe
from core.preprocessing.nyul import LANDMARKS, Nyul
from core.preprocessing.preprocess import Preprocess


class Normalizer:
    """A per-case intensity/geometry transform bound to one recipe (inplane, N4, Nyúl, norm). `build`
    constructs one from a DataCfg's preprocessing fields and calls `apply_case` per subject — so the
    store never imports `preprocess_case` directly. The Nyúl STANDARD (a normalization axis fit to
    reference data) is fit/loaded via the staticmethods `fit_standard`/`load_standard`."""

    def __init__(self, recipe: Recipe | None = None, nyul_standard: Any = None) -> None:
        self.recipe = recipe or Recipe()
        self.nyul_standard = nyul_standard

    def apply_case(self, case: Path, loader: Any) -> dict[str, Any]:
        """Consolidate ONE raw case to the recipe's processed arrays (resample [+N4] [+Nyúl] + norm)."""
        r = self.recipe
        return Preprocess.preprocess_case(case, target_inplane=r.inplane, loader=loader,
                               n4=r.n4, n4_params=r.n4_params,
                               nyul_standard=self.nyul_standard if r.nyul else None, norm=r.norm)

    @staticmethod
    def ref_path() -> Path:
        return Reference.reference_dir() / "nyul.yaml"

    @staticmethod
    def load_standard() -> "np.ndarray | None":
        """The fitted Nyúl standard from reference/nyul.yaml, or None if absent (then nyul can't run)."""
        v = Reference().get("nyul", "standard")
        return np.asarray(v, dtype=np.float64) if v is not None else None

    @staticmethod
    def fit_standard(  # pragma: no cover  reads real adapter NIfTI + writes reference/nyul.yaml
        names: list[str] | None = None, inplane: float = DEFAULT_INPLANE,
        per_dataset: int = 40,
    ) -> "np.ndarray":
        """Fit the Nyúl standard landmark scale from the cohort (resampled, pre-z-score images) and write
        it to reference/nyul.yaml with provenance. Samples up to per_dataset subjects/dataset (landmarks
        are stable). The standard is a normalization axis -> reference data, fit once, applied per case."""
        names = SOURCE_DATASETS if names is None else names
        rows = []
        for name in names:
            adapter = AdapterRegistry.get_adapter(name)
            for case in adapter.cases()[:per_dataset]:
                d = adapter.load_ed_es(case)
                if Phase.ED not in d:
                    continue
                img, _ = Preprocess.resample_inplane(d[Phase.ED]["img"], d["spacing"], inplane, is_mask=False)
                rows.append(Nyul.image_landmarks(img))
        std = Nyul.fit_standard(np.stack(rows))
        p = Normalizer.ref_path(); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("# Nyúl standard landmark scale (harmonization qfz) — fit by Normalizer.fit_standard\n"
                     + OmegaConf.to_yaml(OmegaConf.create({"nyul": {"standard": {
                         "value": [round(float(v), 5) for v in std], "landmarks": list(LANDMARKS),
                         "source": "computed", "based_on": f"resampled ED, {names}, per<={per_dataset}, n={len(rows)}",
                         "extracted_by": "computed", "verified": True}}})))
        return std
