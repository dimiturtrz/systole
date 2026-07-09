"""bSSFP signal physics (core.data.dynamic.mri_physics). This is pure math over the literature
TISSUE / TISSUE_RANGE / SAR tables, so it's tested properly: the closed-form signal equation, the
per-tissue relaxation bands, physics-derived cine acquisition (TR/TE/flip + SAR cap), and the
per-sample heart-tissue sampler. Equivalence-classes: signal monotonicities, T1/T2 band membership,
SAR flip ceiling, acquisition override-vs-derived, oxygenation split, spread endpoints."""
import math

import torch

from core.data.dynamic.mri_physics import (
    SAR_FLIP_CAP,
    TISSUE,
    TISSUE_RANGE,
    TR_RANGE_MS,
    MriPhysics,
)


def test_bssfp_matches_closed_form():
    """S = PD·sinα·(1−E1) / (1−(E1−E2)cosα − E1·E2) evaluated against a hand-computed point."""
    t1, t2, pd, tr = 1000.0, 200.0, 0.9, 3.0
    flip = 50 * math.pi / 180
    e1, e2 = math.exp(-tr / t1), math.exp(-tr / t2)
    a = math.sin(flip)
    c = math.cos(flip)
    expected = pd * a * (1 - e1) / (1 - (e1 - e2) * c - e1 * e2)
    got = MriPhysics.bssfp_signal(*(torch.tensor(x) for x in (t1, t2, pd, tr, flip)))
    assert abs(float(got) - expected) < 1e-6


def test_bssfp_zero_flip_is_zero():
    """sin(0)=0 -> no transverse signal at 0 flip (boundary)."""
    s = MriPhysics.bssfp_signal(*(torch.tensor(x) for x in (1000.0, 200.0, 1.0, 3.0, 0.0)))
    assert abs(float(s)) < 1e-9


def test_bssfp_scales_with_pd():
    """Signal is linear in proton density (PD is a prefactor)."""
    args = (torch.tensor(1000.0), torch.tensor(200.0))
    tr, flip = torch.tensor(3.0), torch.tensor(0.5)
    s1 = MriPhysics.bssfp_signal(*args, torch.tensor(0.5), tr, flip)
    s2 = MriPhysics.bssfp_signal(*args, torch.tensor(1.0), tr, flip)
    assert abs(float(s2) - 2.0 * float(s1)) < 1e-6


def test_bssfp_blood_brighter_than_myo():
    """The contrast the sequence exists for: long-T2 blood outshines short-T2 myocardium in bSSFP."""
    bt1, bt2, bpd = MriPhysics._params("blood", 1.5)
    mt1, mt2, mpd = MriPhysics._params("myocardium", 1.5)
    tr, flip = torch.tensor(3.0), torch.tensor(50 * math.pi / 180)
    sb = MriPhysics.bssfp_signal(torch.tensor(bt1), torch.tensor(bt2), torch.tensor(bpd), tr, flip)
    sm = MriPhysics.bssfp_signal(torch.tensor(mt1), torch.tensor(mt2), torch.tensor(mpd), tr, flip)
    assert float(sb) > float(sm)


def test_banding_unity_at_passband():
    """Banding factor normalizes to 1 at dphi=0 (on-resonance passband)."""
    b = MriPhysics.banding(torch.tensor(200.0), torch.tensor(3.0), torch.tensor(0.0))
    assert abs(float(b) - 1.0) < 1e-4


def test_banding_deeper_dip_for_long_t2():
    """Long-T2 (blood) bands harder than short-T2 (myo) at the same off-resonance."""
    dphi = torch.tensor(math.pi)                       # deepest dip
    tr = torch.tensor(3.0)
    long_t2 = MriPhysics.banding(torch.tensor(250.0), tr, dphi)     # blood
    short_t2 = MriPhysics.banding(torch.tensor(40.0), tr, dphi)     # myo
    assert float(long_t2) < float(short_t2)


def test_params_picks_nearest_field():
    """_params snaps to the nearest tabulated field strength."""
    assert MriPhysics._params("blood", 1.4) == TISSUE["blood"][1.5]
    assert MriPhysics._params("blood", 2.9) == TISSUE["blood"][3.0]


def test_tissue_range_bounds_ordered_and_positive():
    """Every ((T1lo,T1hi),(T2lo,T2hi)) band is ordered lo<=hi with positive relaxation times — the
    invariant the per-sample sampler relies on (draws uniformly in [lo,hi])."""
    for name in TISSUE_RANGE:
        for field in (1.5, 3.0):
            (t1lo, t1hi), (t2lo, t2hi) = MriPhysics.tissue_range(name, field)
            assert 0 < t1lo <= t1hi and 0 < t2lo <= t2hi


def test_tissue_range_field_shifts_blood_t1_up():
    """Higher field lengthens T1: the blood T1 band at 3T sits above the 1.5T band (cross-vendor axis)."""
    (lo15, hi15), _ = MriPhysics.tissue_range("blood", 1.5)
    (lo30, hi30), _ = MriPhysics.tissue_range("blood", 3.0)
    assert lo30 > lo15 and hi30 > hi15


def test_derive_flip_range_within_sar_cap():
    """Derived cine flip band stays under the field's SAR ceiling, lo<=hi, and is non-degenerate."""
    for field in (1.5, 3.0):
        lo, hi = MriPhysics.derive_flip_range(field)
        assert 1.0 <= lo <= hi <= SAR_FLIP_CAP[field]


def test_derive_acquisition_flip_capped_by_sar():
    """derive_acquisition returns the SAR-capped contrast-optimal flip; TR mid-band, TE=TR/2."""
    tr, te, flip = MriPhysics.derive_acquisition(3.0)
    assert TR_RANGE_MS[0] <= tr <= TR_RANGE_MS[1]
    assert abs(te - tr / 2.0) < 1e-9
    assert flip <= SAR_FLIP_CAP[3.0]


def test_derive_acquisition_contrast_optimal_in_flip_range():
    """The canonical point flip sits inside the domain-randomization band (both from the same curve)."""
    _, _, flip = MriPhysics.derive_acquisition(1.5)
    lo, hi = MriPhysics.derive_flip_range(1.5)
    assert lo <= flip <= hi


def test_acquisition_for_no_ref_equals_derived():
    """With no reference override, acquisition_for == the pure physics derivation."""
    assert MriPhysics.acquisition_for("Siemens", 1.5, ref=None) == MriPhysics.derive_acquisition(1.5)


def test_acquisition_for_ref_overrides_derived():
    """A verified reference leaf replaces the derived value per (vendor, field)."""
    class FakeRef:
        def get(self, *keys):
            return {"tr_ms": 3.2, "te_ms": 1.6, "flip_deg_1p5t": 65.0, "flip_deg_3t": 45.0}.get(keys[-1])

    tr, te, flip = MriPhysics.acquisition_for("GE", 1.5, ref=FakeRef())
    assert (tr, te, flip) == (3.2, 1.6, 65.0)


def test_acquisition_for_ref_partial_override_falls_back():
    """A reference that only supplies some keys keeps the derived value for the missing ones."""
    d_tr, d_te, _ = MriPhysics.derive_acquisition(1.5)

    class PartialRef:
        def get(self, *keys):
            return 70.0 if keys[-1] == "flip_deg_1p5t" else None

    tr, te, flip = MriPhysics.acquisition_for("GE", 1.5, ref=PartialRef())
    assert (tr, te, flip) == (d_tr, d_te, 70.0)


def test_blood_classes_are_the_cavities():
    """Blood = RV cavity (1) + LV cavity (3); myo (2) and bg (0) excluded."""
    assert MriPhysics.blood_classes(4) == [1, 3]
    assert MriPhysics.blood_classes(2) == [1]           # class 3 out of range


def test_tissue_params_length_and_heart_mapping():
    """tissue_params has length n_classes + n_bg_tiers; heart labels map via _HEART."""
    t1, t2, pd = MriPhysics.tissue_params(4, 3, 1.5, "cpu")
    assert t1.shape == t2.shape == pd.shape == (4 + 3,)
    blood = MriPhysics._params("blood", 1.5)
    assert float(t1[1]) == blood[0] and float(t1[3]) == blood[0]     # RV + LV cav = blood


def test_tissue_params_bg_tiers_are_distinct():
    """Background tiers interpolate along the ladder -> K tiers give K distinct T1 values (no collision)."""
    t1, _, _ = MriPhysics.tissue_params(4, 4, 1.5, "cpu")
    bg = t1[4:]
    assert len(set(round(float(v), 3) for v in bg)) == 4


def test_named_tissue_params_orders_by_name():
    """Explicit-name builder returns params in the given order."""
    names = ["fat", "lung"]
    t1, t2, pd = MriPhysics.named_tissue_params(names, 1.5, "cpu")
    for i, nm in enumerate(names):
        row = (float(t1[i]), float(t2[i]), float(pd[i]))
        assert all(abs(a - b) < 1e-3 for a, b in zip(row, MriPhysics._params(nm, 1.5), strict=True))


def test_sample_heart_tissue_spread_zero_is_identity():
    """spread=0 leaves the point values untouched (sampling off)."""
    t1 = torch.tensor([[0.0, 1500.0, 1000.0, 1500.0]])
    t2 = torch.tensor([[0.0, 250.0, 40.0, 250.0]])
    fi = torch.tensor([0], dtype=torch.long)
    o1, o2 = MriPhysics.sample_heart_tissue(t1, t2, fi, (1.5,), 4, spread=0.0)
    assert torch.allclose(o1, t1) and torch.allclose(o2, t2)


def test_sample_heart_tissue_full_spread_lands_in_band():
    """spread=1 draws each heart class fully inside its literature TISSUE_RANGE band at that field."""
    torch.manual_seed(0)
    b = 64
    t1 = torch.zeros(b, 4)
    t2 = torch.zeros(b, 4)
    fi = torch.zeros(b, dtype=torch.long)
    o1, o2 = MriPhysics.sample_heart_tissue(t1, t2, fi, (1.5,), 4, spread=1.0)
    for c, tissue in ((1, "blood"), (2, "myocardium"), (3, "blood")):
        (t1lo, t1hi), (t2lo, t2hi) = MriPhysics.tissue_range(tissue, 1.5)
        assert (o1[:, c] >= t1lo - 1e-3).all() and (o1[:, c] <= t1hi + 1e-3).all()
        assert (o2[:, c] >= t2lo - 1e-3).all() and (o2[:, c] <= t2hi + 1e-3).all()


def test_sample_heart_tissue_oxygenation_splits_t2():
    """LV cav (3, oxygenated) draws from the UPPER T2 half, RV cav (1, deoxy) from the LOWER half."""
    torch.manual_seed(1)
    b = 200
    t1 = torch.zeros(b, 4)
    t2 = torch.zeros(b, 4)
    fi = torch.zeros(b, dtype=torch.long)
    _, o2 = MriPhysics.sample_heart_tissue(t1, t2, fi, (1.5,), 4, spread=1.0)
    (_, _), (t2lo, t2hi) = MriPhysics.tissue_range("blood", 1.5)
    mid = 0.5 * (t2lo + t2hi)
    assert (o2[:, 3] >= mid - 1e-3).all()               # LV cav in upper half
    assert (o2[:, 1] <= mid + 1e-3).all()               # RV cav in lower half


def test_sample_heart_tissue_ignores_out_of_range_class():
    """A heart label >= n_classes is skipped (n_classes=2 keeps only RV cav)."""
    torch.manual_seed(2)
    t1 = torch.zeros(4, 2)
    t2 = torch.zeros(4, 2)
    fi = torch.zeros(4, dtype=torch.long)
    o1, o2 = MriPhysics.sample_heart_tissue(t1, t2, fi, (1.5,), 2, spread=1.0)
    assert (o1[:, 1] > 0).all() and (o2[:, 1] > 0).all()   # RV cav (blood) filled, no index error
