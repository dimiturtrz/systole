"""Golden-output safety net for the synth painter (bd yxg2).

Fixed-seed `synthesize_from_labels` on a fixed mask + a config that exercises nearly every optional
physical stage (deform, procedural bg, tissue-spread, jitter, flow, B0 banding, trabecular PV, inflow,
partial-volume blur, k-space PSF, bias field, blur, Rician noise). Locks the output against a committed
reference so the painter refactors (bd je13 micro-helper DRY, bd 3xzr stage-list) are provably behavior-
identical — any stage reorder or changed RNG-draw order shifts the sampled pixels and fails here.

Regenerate the fixture (only when a KNOWN intentional physics change lands) with the builders below +
`torch.manual_seed(0)`, dumping shape/mean/std/min/max/sig/mask to fixtures/synth_golden.json.
"""
import json
from pathlib import Path

import torch

from core.data.dynamic.synth import SynthCfg, SynthPainter

_N_CLASSES = 4
_ATOL = 1e-4
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


def test_synth_is_deterministic_under_fixed_seed():
    img1, mask1 = _synth()
    img2, mask2 = _synth()
    assert torch.equal(img1, img2)
    assert torch.equal(mask1, mask2)


def test_synth_matches_golden_reference():
    ref = json.loads(_FIXTURE.read_text())
    img, mask = _synth()

    assert list(img.shape) == ref["shape"]
    assert abs(float(img.mean()) - ref["mean"]) < _ATOL
    assert abs(float(img.std()) - ref["std"]) < _ATOL
    assert abs(float(img.min()) - ref["min"]) < _ATOL
    assert abs(float(img.max()) - ref["max"]) < _ATOL

    sig = img.flatten()[::ref["sig_stride"]]
    assert torch.allclose(sig, torch.tensor(ref["sig"]), atol=_ATOL)

    assert int(mask.sum()) == ref["mask_sum"]
    assert sorted(int(u) for u in mask.unique()) == ref["mask_unique"]
