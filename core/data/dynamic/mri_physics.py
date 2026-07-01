"""bSSFP signal physics for synthetic contrast (bd cardiac-seg-276).

Contrast comes from the actual MRI signal equation: a tissue's intensity = its relaxation params
(T1/T2/PD) through the pulse sequence. Cardiac cine is balanced SSFP, whose steady-state on-resonance
signal is a closed form of (T1, T2, TR, flip) — no Bloch integration needed.

Why physical beats statistical jitter for cross-vendor: scanner/protocol differences ARE physical
(TR, flip, field strength). Sweeping those sweeps the contrast the way a real protocol change does —
all tissues move TOGETHER along the physical manifold — and FIELD strength (1.5T vs 3T) changes T1/T2,
which is the dominant cross-vendor axis in this multi-field cohort.

Literature T1/T2/PD (ms, ms, a.u.): Stanisz et al., MRM 2005; Bojorquez et al., MRI 2017 (review).
Approximate — fine for domain randomization.
"""
from __future__ import annotations

import torch

# tissue -> field (Tesla) -> (T1 ms, T2 ms, PD). Two fields = the cross-vendor relaxation axis.
TISSUE: dict[str, dict[float, tuple[float, float, float]]] = {
    "blood":      {1.5: (1540.0, 250.0, 0.95), 3.0: (1650.0, 150.0, 0.95)},  # long T2 -> bright in bSSFP
    "myocardium": {1.5: (1030.0,  40.0, 0.80), 3.0: (1200.0,  45.0, 0.80)},
    "fat":        {1.5: (290.0,  165.0, 1.00), 3.0: (370.0,  130.0, 1.00)},  # short T1 -> bright
    "lung":       {1.5: (1300.0,  50.0, 0.15), 3.0: (1550.0,  45.0, 0.15)},  # low PD (air) -> dark
    "liver":      {1.5: (580.0,   45.0, 0.70), 3.0: (810.0,   34.0, 0.70)},
    "muscle":     {1.5: (1010.0,  35.0, 0.75), 3.0: (1420.0,  32.0, 0.75)},
}
FIELDS: tuple[float, ...] = (1.5, 3.0)

# canonical heart label -> tissue: 1=RV cavity (blood), 2=myocardium, 3=LV cavity (blood); 0=bg fallback.
_HEART = {1: "blood", 2: "myocardium", 3: "blood"}
# background intensity ladder dark -> bright; bg tiers INTERPOLATE params along it (no collisions, any K).
_BG_LADDER = ("lung", "liver", "muscle", "fat")


# Typical cine bSSFP SAX acquisition per vendor: (TR ms, flip deg). Approximate vendor protocols
# (Siemens TrueFISP / Philips bFFE / GE FIESTA / Canon TrueSSFP) — the calibration axis. Overridable
# from reference/acquisition.yaml (measured/study values) so synth contrast matches the real scanner.
_ACQ: dict[str, tuple[float, float]] = {
    "Siemens": (3.0, 52.0),
    "Philips": (3.0, 45.0),
    "GE":      (3.4, 45.0),
    "Canon":   (3.2, 50.0),
}
_ACQ_DEFAULT = (3.1, 48.0)


def acquisition_for(vendor: str | None, ref=None) -> tuple[float, float]:
    """(TR ms, flip deg) for a vendor's typical cine bSSFP — the acquisition to calibrate synth contrast
    to. Prefers reference/acquisition.yaml (ref.get('acquisition', vendor, 'tr_ms'/'flip_deg')); else the
    typical per-vendor constant; else the generic default. `ref` = a core.data.static.reference.Reference (optional)."""
    tr = fl = None
    if ref is not None and vendor is not None:
        tr = ref.get("acquisition", vendor, "tr_ms")
        fl = ref.get("acquisition", vendor, "flip_deg")
    d = _ACQ.get(vendor or "", _ACQ_DEFAULT)
    return (float(tr) if tr is not None else d[0], float(fl) if fl is not None else d[1])


def blood_classes(n_classes: int) -> list[int]:
    """Label indices whose tissue is blood (RV + LV cavities) — the pools that show flow signal
    variation in cine, so they get the extra `flow` texture."""
    return [c for c in range(n_classes) if _HEART.get(c) == "blood"]


def bssfp_signal(T1: torch.Tensor, T2: torch.Tensor, PD: torch.Tensor,
                 TR: torch.Tensor, flip: torch.Tensor) -> torch.Tensor:
    """Balanced-SSFP steady-state ON-RESONANCE signal (Freeman–Hill). T1/T2/TR in ms, flip in radians;
    broadcasting tensors. S = PD·sinα·(1−E1) / (1−(E1−E2)cosα − E1·E2), E1=exp(−TR/T1), E2=exp(−TR/T2).
    The passband (max) value; off-resonance banding multiplies this — see `banding`."""
    e1 = torch.exp(-TR / T1)
    e2 = torch.exp(-TR / T2)
    s, c = torch.sin(flip), torch.cos(flip)
    return PD * s * (1.0 - e1) / (1.0 - (e1 - e2) * c - e1 * e2)


def banding(T2: torch.Tensor, TR: torch.Tensor, dphi) -> torch.Tensor:
    """bSSFP off-resonance banding factor in (0, 1], normalized to 1 at the passband (dphi=0). dphi =
    off-resonance precession per TR (rad, = 2π·Δf·TR). ratio = (1−E2) / |1−E2·e^{iφ}|
    = (1−E2)/√(1−2E2cosφ+E2²): =1 at φ=0, dips toward φ=±π, and the dip is DEEPER for long-T2 tissue
    (E2→1) — so blood bands hard, short-T2 myocardium barely. Multiplies the on-resonance signal."""
    e2 = torch.exp(-TR / T2)
    return (1.0 - e2) / torch.sqrt(1.0 - 2.0 * e2 * torch.cos(dphi) + e2 ** 2).clamp_min(1e-6)


def _params(name: str, field: float) -> tuple[float, float, float]:
    """(T1,T2,PD) for a tissue at the nearest tabulated field strength."""
    table = TISSUE[name]
    f = min(table, key=lambda k: abs(k - field))
    return table[f]


def tissue_params(n_classes: int, n_bg_tiers: int, field: float,
                  device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(T1, T2, PD) tensors length n_classes + n_bg_tiers at the given field. Heart labels via _HEART;
    background tiers INTERPOLATE params smoothly along _BG_LADDER (dark->bright) so K tiers give K
    distinct backgrounds (no round() collisions). Index 0 (bg) = muscle fallback."""
    rows: list[tuple[float, float, float]] = [_params(_HEART.get(c, "muscle"), field) for c in range(n_classes)]
    ladder = [_params(nm, field) for nm in _BG_LADDER]
    for t in range(n_bg_tiers):
        pos = (t / max(1, n_bg_tiers - 1)) * (len(ladder) - 1)      # 0 .. len-1
        lo = int(pos)
        hi = min(lo + 1, len(ladder) - 1)
        w = pos - lo                                                # interpolation weight
        rows.append(tuple(ladder[lo][i] * (1 - w) + ladder[hi][i] * w for i in range(3)))
    t1 = torch.tensor([r[0] for r in rows], device=device)
    t2 = torch.tensor([r[1] for r in rows], device=device)
    pd = torch.tensor([r[2] for r in rows], device=device)
    return t1, t2, pd
