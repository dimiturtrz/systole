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

import math

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
_FIELD_1P5T = 1.5                            # tesla; the low-field column (else use the 3T flip value)

# tissue -> field -> ((T1_min,T1_max),(T2_min,T2_max)) ms: the literature SPREAD (inter-subject +
# inter-study/method), for per-sample physical sampling (bd 04bh). Prefer in-vivo mapping over the
# in-vitro Stanisz table for myo/blood. Full sourcing + confidence flags:
# research/deep_dives/2026-07-08_tissue_relaxation_ranges.md. TISSUE points sit inside these bands.
TISSUE_RANGE: dict[str, dict[float, tuple[tuple[float, float], tuple[float, float]]]] = {
    "blood":      {1.5: ((1350.0, 1550.0), (200.0, 290.0)), 3.0: ((1550.0, 2100.0), (100.0, 165.0))},
    "myocardium": {1.5: (( 950.0, 1050.0), ( 45.0,  56.0)), 3.0: ((1150.0, 1300.0), ( 45.0,  52.0))},
    "fat":        {1.5: (( 250.0,  380.0), (100.0, 165.0)), 3.0: (( 370.0,  450.0), (100.0, 220.0))},
    "lung":       {1.5: (( 800.0, 1300.0), ( 40.0,  50.0)), 3.0: ((1000.0, 1400.0), ( 30.0,  50.0))},
    "liver":      {1.5: (( 500.0,  650.0), ( 40.0, 102.0)), 3.0: (( 700.0,  900.0), ( 30.0,  50.0))},
    "muscle":     {1.5: (( 870.0, 1100.0), ( 35.0,  50.0)), 3.0: ((1300.0, 1450.0), ( 30.0,  50.0))},
}

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
TR_RANGE_MS = (2.7, 3.5)                     # cited cross-vendor cine bSSFP TR band (mriquestions; SCMR
#                                             family, PMC7038611). Sampled per synth sample (not a point +
#                                             magic jitter); TE = TR/2. The gradient/banding-limited range.
SAR_FLIP_CAP = {1.5: 80.0, 3.0: 50.0}       # SAR-limited max flip (deg); SAR ~ B0^2 (PubMed 26509846)
_TR_MID = sum(TR_RANGE_MS) / 2.0            # representative TR for the canonical (point) derivation


_CAVITY_O2 = {1: "low", 3: "high"}       # RV cavity = deoxygenated (short T2) / LV cavity = oxygenated (long T2)


class MriPhysics:
    """bSSFP signal physics for synthetic contrast (bd cardiac-seg-276): the signal equation, physics-
    derived cine acquisition (TR/TE/flip), and per-tissue relaxation params/sampling. All staticmethods —
    stateless functions over the module-level TISSUE / TISSUE_RANGE / SAR tables."""

    @staticmethod
    def _contrast_optimal_flip(field: float, tr_ms: float) -> float:
        """Flip (deg, integer sweep) maximizing |S_blood - S_myo| bSSFP contrast at `field`, from the TISSUE
        T1/T2 table. Cine targets blood-myocardium contrast; this DERIVES the flip that maximizes it rather
        than quoting a routine-protocol value (which is SNR-, not contrast-, optimized)."""
        bt1, bt2, bpd = MriPhysics._params("blood", field)
        mt1, mt2, mpd = MriPhysics._params("myocardium", field)
        a = torch.arange(1.0, 91.0)
        rad = a * math.pi / 180.0
        tr = torch.tensor(tr_ms)
        sb = MriPhysics.bssfp_signal(torch.tensor(bt1), torch.tensor(bt2), torch.tensor(bpd), tr, rad)
        sm = MriPhysics.bssfp_signal(torch.tensor(mt1), torch.tensor(mt2), torch.tensor(mpd), tr, rad)
        return float(a[(sb - sm).abs().argmax()])

    @staticmethod
    def derive_acquisition(field: float) -> tuple[float, float, float]:
        """(TR ms, TE ms, flip deg) DERIVED for cine bSSFP at `field`: TR = mid of the cited band, TE=TR/2,
        flip = blood-myo contrast-optimal capped by SAR. Reproducible from the signal equation + TISSUE — no
        magic constants (this is the canonical POINT; synth samples TR/flip across their ranges)."""
        f = min(SAR_FLIP_CAP, key=lambda x: abs(x - float(field))) if field else 1.5
        flip = min(MriPhysics._contrast_optimal_flip(f, _TR_MID), SAR_FLIP_CAP[f])
        return (_TR_MID, _TR_MID / 2.0, flip)

    @staticmethod
    def derive_flip_range(field: float) -> tuple[float, float]:
        """Cine flip RANGE (deg) for DOMAIN RANDOMIZATION, DERIVED from the contrast curve: the FWHM band
        where blood-myo bSSFP contrast is >= half its peak, capped by SAR. Bounds come from the physics (the
        half-maximum convention on the contrast curve + the SAR ceiling) — no arbitrary fraction of the cap.
        Sampling across this band gives contrast DIVERSITY (measured: the single contrast-optimal point
        trains WORSE — fidelity != training value, bd 276). Field-driven; contrast-optimal sits inside it."""
        f = min(SAR_FLIP_CAP, key=lambda x: abs(x - float(field))) if field else 1.5
        bt1, bt2, bpd = MriPhysics._params("blood", f)
        mt1, mt2, mpd = MriPhysics._params("myocardium", f)
        a = torch.arange(1.0, 91.0)
        rad = a * math.pi / 180.0
        tr = torch.tensor(_TR_MID)
        sb = MriPhysics.bssfp_signal(torch.tensor(bt1), torch.tensor(bt2), torch.tensor(bpd), tr, rad)
        sm = MriPhysics.bssfp_signal(torch.tensor(mt1), torch.tensor(mt2), torch.tensor(mpd), tr, rad)
        c = (sb - sm).abs()
        band = a[c >= 0.5 * c.max()]                             # FWHM of the contrast curve (half-max convention)
        return (float(band.min()), min(float(band.max()), SAR_FLIP_CAP[f]))

    @staticmethod
    def acquisition_for(vendor: str | None, field: float = 1.5, ref=None) -> tuple[float, float, float]:
        """(TR ms, TE ms, flip deg) for a machine = (vendor, field) cine bSSFP. Base = the physics
        DERIVATION (derive_acquisition, field-driven). A verified reference/ leaf overrides it per vendor
        (e.g. a DICOM-mined measurement): tr_ms / te_ms / flip_deg_1p5t / flip_deg_3t. `ref` = a
        core.data.static.reference.Reference (optional)."""
        tr, te, fl = MriPhysics.derive_acquisition(field)
        if ref is not None and vendor is not None:
            f = min(SAR_FLIP_CAP, key=lambda x: abs(x - float(field))) if field else 1.5
            o_tr = ref.get("acquisition", vendor, "tr_ms")
            o_te = ref.get("acquisition", vendor, "te_ms")
            o_fl = ref.get("acquisition", vendor, "flip_deg_1p5t" if f == _FIELD_1P5T else "flip_deg_3t")
            tr = float(o_tr) if o_tr is not None else tr
            te = float(o_te) if o_te is not None else te
            fl = float(o_fl) if o_fl is not None else fl
        return (tr, te, fl)

    @staticmethod
    def blood_classes(n_classes: int) -> list[int]:
        """Label indices whose tissue is blood (RV + LV cavities) — the pools that show flow signal
        variation in cine, so they get the extra `flow` texture."""
        return [c for c in range(n_classes) if _HEART.get(c) == "blood"]

    @staticmethod
    def bssfp_signal(T1: torch.Tensor, T2: torch.Tensor, PD: torch.Tensor,
                     TR: torch.Tensor, flip: torch.Tensor) -> torch.Tensor:
        """Balanced-SSFP steady-state ON-RESONANCE signal (Freeman–Hill). T1/T2/TR in ms, flip in radians;
        broadcasting tensors. S = PD·sinα·(1−E1) / (1−(E1−E2)cosα − E1·E2), E1=exp(−TR/T1), E2=exp(−TR/T2).
        The passband (max) value; off-resonance banding multiplies this — see `banding`."""
        e1 = torch.exp(-TR / T1)
        e2 = torch.exp(-TR / T2)
        s, c = torch.sin(flip), torch.cos(flip)
        return PD * s * (1.0 - e1) / (1.0 - (e1 - e2) * c - e1 * e2)

    @staticmethod
    def banding(T2: torch.Tensor, TR: torch.Tensor, dphi) -> torch.Tensor:
        """bSSFP off-resonance banding factor in (0, 1], normalized to 1 at the passband (dphi=0). dphi =
        off-resonance precession per TR (rad, = 2π·Δf·TR). ratio = (1−E2) / |1−E2·e^{iφ}|
        = (1−E2)/√(1−2E2cosφ+E2²): =1 at φ=0, dips toward φ=±π, and the dip is DEEPER for long-T2 tissue
        (E2→1) — so blood bands hard, short-T2 myocardium barely. Multiplies the on-resonance signal."""
        e2 = torch.exp(-TR / T2)
        return (1.0 - e2) / torch.sqrt(1.0 - 2.0 * e2 * torch.cos(dphi) + e2 ** 2).clamp_min(1e-6)

    @staticmethod
    def _params(name: str, field: float) -> tuple[float, float, float]:
        """(T1,T2,PD) for a tissue at the nearest tabulated field strength."""
        table = TISSUE[name]
        f = min(table, key=lambda k: abs(k - field))
        return table[f]

    @staticmethod
    def tissue_range(name: str, field: float) -> tuple[tuple[float, float], tuple[float, float]]:
        """((T1_min,T1_max),(T2_min,T2_max)) ms for a tissue at the nearest tabulated field — the literature
        spread the per-sample sampler (bd 04bh) draws over. See TISSUE_RANGE."""
        table = TISSUE_RANGE[name]
        f = min(table, key=lambda k: abs(k - field))
        return table[f]

    @staticmethod
    def sample_heart_tissue(t1: torch.Tensor, t2: torch.Tensor, fi: torch.Tensor,  # noqa: PLR0913
                            fields: tuple[float, ...], n_classes: int, spread: float):
        """Per-sample redraw of HEART-class (blood/myo) T1/T2 from the literature TISSUE_RANGE band, lerped
        from the current point value by `spread` (0=point/off, 1=full uniform band). Physically-constrained
        breadth (UltimateSynth) in place of decorrelated jitter: contrast then flows through bssfp_signal, so
        the tissues move together the way a real protocol/subject change does. Blood cavities split by
        oxygenation — LV (long T2, upper half of the band), RV (short T2, lower half). t1/t2 [B, n_paint];
        fi [B] long, field index into `fields`. Returns fresh (t1, t2)."""
        dev = t1.device
        t1, t2 = t1.clone(), t2.clone()
        b = t1.shape[0]
        for c, tissue in _HEART.items():
            if c >= n_classes:
                continue
            bands = [MriPhysics.tissue_range(tissue, float(f)) for f in fields]   # per field ((t1..),(t2..))
            t1lo = torch.tensor([bd[0][0] for bd in bands], device=dev)[fi]       # [B], per-sample field band
            t1hi = torch.tensor([bd[0][1] for bd in bands], device=dev)[fi]
            t2lo = torch.tensor([bd[1][0] for bd in bands], device=dev)[fi]
            t2hi = torch.tensor([bd[1][1] for bd in bands], device=dev)[fi]
            if tissue == "blood":                                                # oxygenation splits the T2 band
                mid = 0.5 * (t2lo + t2hi)
                if _CAVITY_O2.get(c) == "high":
                    t2lo = mid
                elif _CAVITY_O2.get(c) == "low":
                    t2hi = mid
            s1 = t1lo + torch.rand(b, device=dev) * (t1hi - t1lo)
            s2 = t2lo + torch.rand(b, device=dev) * (t2hi - t2lo)
            t1[:, c] = t1[:, c] + spread * (s1 - t1[:, c])
            t2[:, c] = t2[:, c] + spread * (s2 - t2[:, c])
        return t1, t2

    @staticmethod
    def named_tissue_params(names: list[str], field: float, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(T1,T2,PD) tensors for an EXPLICIT list of tissue names (each a key of TISSUE) at `field`. Generic
        builder for sources that assign every class a specific tissue — e.g. MRXCAT whole-FOV paints classes
        0..7 = air/blood/myo/blood/lung/liver/muscle/fat — instead of the heart+bg-ladder default of
        `tissue_params`. Index order = the given name order (so it lines up with the class ids)."""
        rows = [MriPhysics._params(nm, field) for nm in names]
        t1 = torch.tensor([r[0] for r in rows], device=device)
        t2 = torch.tensor([r[1] for r in rows], device=device)
        pd = torch.tensor([r[2] for r in rows], device=device)
        return t1, t2, pd

    @staticmethod
    def tissue_params(n_classes: int, n_bg_tiers: int, field: float,
                      device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(T1, T2, PD) tensors length n_classes + n_bg_tiers at the given field. Heart labels via _HEART;
        background tiers INTERPOLATE params smoothly along _BG_LADDER (dark->bright) so K tiers give K
        distinct backgrounds (no round() collisions). Index 0 (bg) = muscle fallback."""
        rows: list[tuple[float, float, float]] = [
            MriPhysics._params(_HEART.get(c, "muscle"), field) for c in range(n_classes)
        ]
        ladder = [MriPhysics._params(nm, field) for nm in _BG_LADDER]
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
