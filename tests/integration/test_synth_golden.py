"""Golden-output safety net for the synth painter (bd yxg2 / o9lx) — INTEGRATION tier.

This spans synth ↔ mri_physics ↔ augment ↔ the bg/acq strategy classes, so it lives in integration/,
not unit/: its whole value is the cross-stage COMPOSITION (RNG draw order, stage wiring) that a
per-transform unit test can't see. Two independent jobs, kept separate:

- `test_synth_output_is_well_formed` — ref-INDEPENDENT correctness oracle (z-score => mean~0/std~1,
  labels in-range, finite). Validates the state is well-formed regardless of the frozen fixture.
- `test_synth_matches_golden_reference` — CHARACTERIZATION lock: fixed-seed output vs the committed
  fixture (sampled pixels + summaries). Proves a refactor changed NOTHING at bit fidelity; it does
  NOT prove correctness — that's the well-formed test's job, plus real-data validation elsewhere.

Regenerate the fixture (only for a KNOWN intentional physics change) with:
    uv run pytest tests/integration/test_synth_golden.py --update-golden
which recomputes it through the builders below — no separate generator to drift.
"""
import json
from pathlib import Path

import pytest
import torch

from core.data.dynamic.synth import SynthCfg, SynthPainter

_N_CLASSES = 4
_ATOL = 1e-4
_UNIT_STD = 1.0
_ZSCORE_TOL = 1e-3
_SEG_LABELS = {0, 1, 2, 3}
_FIXTURE = Path(__file__).parent / "fixtures" / "synth_golden.json"


def _golden_mask() -> torch.Tensor:
    m = torch.zeros(2, 24, 24, dtype=torch.long)
    m[:, 6:18, 6:18] = 2          # myo block
    m[:, 9:15, 9:15] = 3          # LV-cav inside
    m[:, 6:11, 18:22] = 1         # RV-cav beside
    return m


def _golden_cfg() -> SynthCfg:
    return SynthCfg(
        deform=0.15, bg={"mode": "procedural", "bg_blobs": 6},
        tissue_spread=0.5, jitter=0.4, texture=0.05, flow=0.1, b0_hz=10.0,
        trabec_lv=0.12, trabec_rv=0.24, inflow=True,
        pv_sigma=0.7, kspace=0.8, bias_strength=0.3, blur=(0.3, 0.6), noise=0.05,
    )


def _synth():
    torch.manual_seed(0)
    return SynthPainter.synthesize_from_labels(_golden_mask(), _golden_cfg(), _N_CLASSES)


def _reference(img: torch.Tensor, mask: torch.Tensor) -> dict:
    """The committed golden payload: coarse summaries + a strided pixel sample + mask invariants."""
    return {
        "shape": list(img.shape),
        "mean": float(img.mean()), "std": float(img.std()),
        "min": float(img.min()), "max": float(img.max()),
        "sig_stride": 61, "sig": img.flatten()[::61].tolist(),
        "mask_sum": int(mask.sum()), "mask_unique": sorted(int(u) for u in mask.unique()),
    }


def test_synth_is_deterministic_under_fixed_seed():
    img1, mask1 = _synth()
    img2, mask2 = _synth()
    assert torch.equal(img1, img2)
    assert torch.equal(mask1, mask2)


def test_synth_output_is_well_formed():
    """Ref-INDEPENDENT correctness oracle — holds by construction, not by the frozen fixture."""
    img, mask = _synth()
    assert torch.isfinite(img).all()                              # no NaN/inf escaped the chain
    assert abs(float(img.mean())) < _ZSCORE_TOL                   # per-sample z-score => zero mean
    assert abs(float(img.std()) - _UNIT_STD) < _ZSCORE_TOL        # ...and unit std
    assert {int(u) for u in mask.unique()} <= _SEG_LABELS         # seg target stays canonical {0,1,2,3}


def test_synth_matches_golden_reference(request):
    img, mask = _synth()
    if request.config.getoption("--update-golden"):
        _FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        _FIXTURE.write_text(json.dumps(_reference(img, mask), indent=2))
        pytest.skip("golden fixture regenerated")

    ref = json.loads(_FIXTURE.read_text())
    assert list(img.shape) == ref["shape"]
    assert abs(float(img.mean()) - ref["mean"]) < _ATOL
    assert abs(float(img.std()) - ref["std"]) < _ATOL
    assert abs(float(img.min()) - ref["min"]) < _ATOL
    assert abs(float(img.max()) - ref["max"]) < _ATOL

    sig = img.flatten()[::ref["sig_stride"]]
    assert torch.allclose(sig, torch.tensor(ref["sig"]), atol=_ATOL)

    assert int(mask.sum()) == ref["mask_sum"]
    assert sorted(int(u) for u in mask.unique()) == ref["mask_unique"]
