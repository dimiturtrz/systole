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
# NB ends at fat, NOT blood: appending a blood endmember was measured to WORSEN synth-real fidelity
# (mean W1 0.515->0.57, bg 0.236->0.295) — bg composition already matches real (bg location ~0.005), so
# the pure-synth gap is NOT background. It's blood-pool over-brightness (LV-cav W1 0.93, ~all location
# 0.91, shape 0.05). See bd 276 / ap6 notes. (measured 2026-07-01)
_BG_LADDER = ("lung", "liver", "muscle", "fat")


# --- cine bSSFP acquisition, DERIVED from physics (not tabulated paper mid-bands) ---
# The numbers are computed with an argument, reproducible from the signal equation + literature T1/T2:
#   TR   = shortest feasible cine bSSFP TR (gradient-slew + banding floor, ~2.7-3ms). A physical floor.
#   TE   = TR/2 (balanced-SSFP symmetric echo). Derived.
#   flip = the flip MAXIMIZING |S_blood - S_myo| bSSFP contrast (the cine/segmentation-relevant contrast)
#          at the given field, from TISSUE T1/T2, CAPPED by the SAR ceiling (SAR ~ B0^2*flip^2 -> ~80deg
#          @1.5T, ~50deg@3T; Wang/Nayak PubMed 26509846). Field-driven (T1/T2 shift with B0), ~vendor-
#          invariant (no per-model cine tables exist; deep-dive 2026-07-01_mri-vendor-acquisition-params).
# NORMALIZED machine dimension: keyed by FIELD (the axis flip actually depends on). Subjects hold the FK
# (vendor/scanner/field_T in the store) and JOIN via acquisition_for; a reference/ override slot lets a
# human replace the derived value with a DICOM-mined per-(vendor,field) measurement later. (bd ex1/276)
TR_MIN_MS = 2.8                              # cine bSSFP TR floor (gradient/banding limited)
SAR_FLIP_CAP = {1.5: 80.0, 3.0: 50.0}       # SAR-limited max flip (deg); SAR ~ B0^2 (PubMed 26509846)


def _contrast_optimal_flip(field: float, tr_ms: float) -> float:
    """Flip (deg, integer sweep) maximizing |S_blood - S_myo| bSSFP contrast at `field`, from the TISSUE
    T1/T2 table. Cine targets blood-myocardium contrast; this DERIVES the flip that maximizes it rather
    than quoting a routine-protocol value (which is SNR-, not contrast-, optimized)."""
    import math
    bt1, bt2, bpd = _params("blood", field)
    mt1, mt2, mpd = _params("myocardium", field)
    a = torch.arange(1.0, 91.0)
    rad = a * math.pi / 180.0
    tr = torch.tensor(tr_ms)
    sb = bssfp_signal(torch.tensor(bt1), torch.tensor(bt2), torch.tensor(bpd), tr, rad)
    sm = bssfp_signal(torch.tensor(mt1), torch.tensor(mt2), torch.tensor(mpd), tr, rad)
    return float(a[(sb - sm).abs().argmax()])


def derive_acquisition(field: float) -> tuple[float, float, float]:
    """(TR ms, TE ms, flip deg) DERIVED for cine bSSFP at `field`: TR floor, TE=TR/2, flip = blood-myo
    contrast-optimal capped by SAR. Reproducible from the signal equation + TISSUE — no magic constants."""
    f = min(SAR_FLIP_CAP, key=lambda x: abs(x - float(field))) if field else 1.5
    flip = min(_contrast_optimal_flip(f, TR_MIN_MS), SAR_FLIP_CAP[f])
    return (TR_MIN_MS, TR_MIN_MS / 2.0, flip)


def derive_flip_range(field: float) -> tuple[float, float]:
    """Physically-plausible cine flip RANGE (deg) for DOMAIN RANDOMIZATION: low-contrast end .. SAR cap.
    Real acquisition flip spans this whole band; sampling across it gives contrast DIVERSITY. Measured:
    collapsing to the single contrast-optimal flip trains WORSE (fidelity != training value — domain
    randomization needs breadth, bd 276). SAR cap is the physical ceiling; the low end ~half of it
    (routine low-contrast cine). Field-driven; contrast-optimal sits inside the band."""
    f = min(SAR_FLIP_CAP, key=lambda x: abs(x - float(field))) if field else 1.5
    hi = SAR_FLIP_CAP[f]
    return (0.5 * hi, hi)                                    # 1.5T ->(40,80), 3T ->(25,50)


def acquisition_for(vendor: str | None, field: float = 1.5, ref=None) -> tuple[float, float, float]:
    """(TR ms, TE ms, flip deg) for a machine = (vendor, field) cine bSSFP. Base = the physics
    DERIVATION (derive_acquisition, field-driven). A verified reference/ leaf overrides it per vendor
    (e.g. a DICOM-mined measurement): tr_ms / te_ms / flip_deg_1p5t / flip_deg_3t. `ref` = a
    core.data.static.reference.Reference (optional)."""
    tr, te, fl = derive_acquisition(field)
    if ref is not None and vendor is not None:
        f = min(SAR_FLIP_CAP, key=lambda x: abs(x - float(field))) if field else 1.5
        o_tr = ref.get("acquisition", vendor, "tr_ms")
        o_te = ref.get("acquisition", vendor, "te_ms")
        o_fl = ref.get("acquisition", vendor, "flip_deg_1p5t" if f == 1.5 else "flip_deg_3t")
        tr = float(o_tr) if o_tr is not None else tr
        te = float(o_te) if o_te is not None else te
        fl = float(o_fl) if o_fl is not None else fl
    return (tr, te, fl)


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
