"""Physics-based synthetic-from-labels generation (cardioseg.data.synth + mri_physics, bd 276).
ONE paint path: bSSFP signal from tissue T1/T2/PD under swept TR/flip/field. Contract: label mask ->
z-scored image painted by tissue physics; deform invents anatomy; partition gives bg real shapes;
target = real labels."""
import math
import torch

from core.hparams import SynthCfg
from cardioseg.data.synth import synthesize_from_labels
from cardioseg.data.mri_physics import bssfp_signal, tissue_params, TISSUE

N = 4  # canonical classes: 0 bg, 1 RV, 2 LV-myo, 3 LV-cav


def _mask(b=2, h=8, w=8):
    """Samples with all 4 classes in horizontal bands."""
    m = torch.zeros(b, h, w, dtype=torch.long)
    for c in range(N):
        m[:, c * (h // N):(c + 1) * (h // N), :] = c
    return m


def _fixed(**kw):
    """A deterministic-contrast SynthCfg: single field, fixed TR/flip, no jitter/texture/corruption —
    so the paint is purely the bSSFP signal (for ordering assertions)."""
    base = dict(synth_p=1.0, deform=0.0, bg_mode="flat", fields=(1.5,), tr_ms=(3.0, 3.0),
                flip_deg=(45.0, 45.0), jitter=0.0, texture=0.0, bias_strength=0.0, blur=(0.0, 0.0),
                noise=0.0, kspace=0.0)
    base.update(kw)
    return SynthCfg(**base)


# --- physics ---
def test_bssfp_blood_brighter_than_myo():
    """The signal equation reproduces the real cue: blood (long T2) brighter than myocardium at cine
    flip angles. Physics, not assumption."""
    t = torch.tensor
    flip, tr = t([45.0 * math.pi / 180]), t([3.0])
    sig = lambda nm: bssfp_signal(t([TISSUE[nm][1.5][0]]), t([TISSUE[nm][1.5][1]]),
                                  t([TISSUE[nm][1.5][2]]), tr, flip)
    assert sig("blood") > sig("myocardium")


def test_tissue_params_field_shifts_and_bg_distinct():
    """Field strength changes T1 (cross-vendor axis); bg tiers are all distinct (interpolated, no
    round() collisions — the fixed bug)."""
    t1_15, _, _ = tissue_params(N, 0, 1.5, "cpu")
    t1_30, _, _ = tissue_params(N, 0, 3.0, "cpu")
    assert not torch.allclose(t1_15, t1_30)                      # field shifts relaxation
    _, t2_bg, _ = tissue_params(N, 6, 1.5, "cpu")
    bg = t2_bg[N:]                                               # the 6 background tiers
    assert bg.unique().numel() == 6                             # all distinct (collision fixed)


# --- generation ---
def test_shape_and_zscore():
    img, msk = synthesize_from_labels(_mask(), SynthCfg(synth_p=1.0, bg_mode="flat"), N)
    assert img.shape == (2, 1, 8, 8) and msk.shape == (2, 8, 8)
    assert torch.allclose(img.mean((1, 2, 3)), torch.zeros(2), atol=1e-4)
    assert torch.allclose(img.std((1, 2, 3)), torch.ones(2), atol=1e-1)


def test_paint_orders_by_physics():
    """Pure bSSFP paint -> blood classes (RV, cav) brighter than the myo region."""
    torch.manual_seed(0)
    img, _ = synthesize_from_labels(_mask(1), _fixed(), N)
    band = lambda c: img[0, 0, c * 2:(c + 1) * 2].mean()
    assert band(1) > band(2) and band(3) > band(2)              # RV,cav (blood) > myo


def test_deform_invents_anatomy():
    m = _mask()
    torch.manual_seed(1)
    _, warped = synthesize_from_labels(m, SynthCfg(synth_p=1.0, deform=0.3, bg_mode="flat"), N)
    assert not torch.equal(warped, m)
    assert set(warped.unique().tolist()).issubset(set(range(N)))
    _, same = synthesize_from_labels(m, SynthCfg(synth_p=1.0, deform=0.0, bg_mode="flat"), N)
    assert torch.equal(same, m)


def test_partition_bg_structures_image_keeps_target():
    """bg_mode='partition' splits the bg by REAL per-slice intensity into tiers -> image bg has
    structure (>1 level), but the TARGET keeps only real labels (bg=0)."""
    Y = _mask()
    real = torch.randn(2, 1, 8, 8)
    cfg = _fixed(bg_mode="partition", bg_tiers=4)
    torch.manual_seed(0)
    img, msk = synthesize_from_labels(Y, cfg, N, real_img=real)
    assert set(msk.unique().tolist()).issubset(set(range(N)))
    assert img[0, 0][Y[0] == 0].unique().numel() > 1


def test_pv_softens_boundaries():
    """pv_sigma>0 blurs the class-MEAN map -> boundary mixes (more distinct levels) vs hard edges."""
    torch.manual_seed(0); hard, _ = synthesize_from_labels(_mask(1), _fixed(pv_sigma=0.0), N)
    torch.manual_seed(0); soft, _ = synthesize_from_labels(_mask(1), _fixed(pv_sigma=1.0), N)
    assert soft[0, 0].unique().numel() > hard[0, 0].unique().numel()


def test_kspace_lowpass_changes_and_keeps_shape():
    torch.manual_seed(0); a, _ = synthesize_from_labels(_mask(), _fixed(kspace=0.0), N)
    torch.manual_seed(0); b, _ = synthesize_from_labels(_mask(), _fixed(kspace=0.5), N)
    assert b.shape == a.shape and torch.isfinite(b).all() and not torch.allclose(a, b)


def test_seed_deterministic():
    cfg, m = SynthCfg(synth_p=1.0, bg_mode="flat"), _mask()
    torch.manual_seed(7); a, ma = synthesize_from_labels(m, cfg, N)
    torch.manual_seed(7); b, mb = synthesize_from_labels(m, cfg, N)
    assert torch.equal(a, b) and torch.equal(ma, mb)


def test_banding_dips_at_pi_deeper_for_blood():
    """Banding factor: 1 at the passband (dphi=0), dips toward dphi=π, and the dip is DEEPER for
    long-T2 blood than short-T2 myocardium (the physical reason blood bands hardest)."""
    from cardioseg.data.mri_physics import banding
    t, tr = torch.tensor, torch.tensor([3.0])
    blood_t2 = t([TISSUE["blood"][1.5][1]]); myo_t2 = t([TISSUE["myocardium"][1.5][1]])
    assert torch.allclose(banding(blood_t2, tr, t([0.0])), torch.ones(1), atol=1e-3)   # passband = 1
    assert banding(blood_t2, tr, t([math.pi])) < 1.0                                    # band dip
    assert banding(blood_t2, tr, t([math.pi])) < banding(myo_t2, tr, t([math.pi]))      # blood bands deeper


def test_acquisition_for_vendor_and_reference_override(tmp_path):
    """Per-vendor cine bSSFP params (the calibration axis): typical constant per vendor, generic
    default for unknown, reference/acquisition.yaml overrides when present."""
    from cardioseg.data.mri_physics import acquisition_for, _ACQ_DEFAULT
    assert acquisition_for("Siemens") == (3.0, 52.0)
    assert acquisition_for("GE") == (3.4, 45.0)
    assert acquisition_for("Nonesuch") == _ACQ_DEFAULT          # unknown -> generic default
    from core.reference import Reference
    (tmp_path / "acquisition.yaml").write_text(
        "acquisition:\n  Siemens:\n"
        "    tr_ms: {value: 2.9, source: study, based_on: x, extracted_by: paper, verified: true}\n"
        "    flip_deg: {value: 60, source: study, based_on: x, extracted_by: paper, verified: true}\n")
    tr, fl = acquisition_for("Siemens", ref=Reference(root=tmp_path))
    assert (tr, fl) == (2.9, 60.0)                              # reference overrides the constant
