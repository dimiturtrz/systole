"""SynthSeg-style synthetic-from-labels generation (cardioseg.data.synth, bd cardiac-seg-bgc).
Contract: label mask -> z-scored image, painted per class. Realistic mode paints around measured real
priors (blood bright / myo dark, the cue that transfers); deform invents anatomy; target = real labels."""
import torch

from core.hparams import SynthCfg
from cardioseg.data.synth import synthesize_from_labels, measure_class_stats

N = 4  # canonical classes: 0 bg, 1 RV, 2 LV-myo, 3 LV-cav


def _mask(b=2, h=8, w=8):
    """Two samples, each with all 4 classes in horizontal bands."""
    m = torch.zeros(b, h, w, dtype=torch.long)
    for c in range(N):
        m[:, c * (h // N):(c + 1) * (h // N), :] = c
    return m


def test_shape_and_zscore():
    img, msk = synthesize_from_labels(_mask(), SynthCfg(synth_p=1.0, realistic=False), N)
    assert img.shape == (2, 1, 8, 8) and msk.shape == (2, 8, 8)
    assert torch.allclose(img.mean((1, 2, 3)), torch.zeros(2), atol=1e-4)   # per-sample mean ~0
    assert torch.allclose(img.std((1, 2, 3)), torch.ones(2), atol=1e-1)     # per-sample std ~1


def test_legacy_painted_per_class():
    """Pure-random mode, no deform/texture/corruption -> each class a single constant, all distinct."""
    cfg = SynthCfg(synth_p=1.0, realistic=False, deform=0.0, sigma=(0.0, 0.0),
                   bias_strength=0.0, blur=(0.0, 0.0), noise=0.0)
    torch.manual_seed(0)
    img, _ = synthesize_from_labels(_mask(), cfg, N)
    assert len(torch.unique(img[0])) == N


def test_measure_class_stats_recovers_priors():
    """Per-class stats of a known image == that image's per-class mean/std."""
    Y = _mask(1, 8, 8)
    X = torch.zeros(1, 1, 8, 8)
    for c in range(N):
        X[0, 0][Y[0] == c] = float(c)                  # class c painted with intensity c
    means, stds = measure_class_stats(X, Y, N)
    assert torch.allclose(means, torch.arange(N, dtype=torch.float), atol=1e-5)
    assert torch.allclose(stds[1:], torch.zeros(N - 1), atol=1e-2)   # constant per class -> ~0 std


def test_realistic_preserves_ordering():
    """With real priors + jitter=0, each class paints at ITS prior mean -> the bright/dark ORDERING the
    net needs is preserved (unlike pure-random)."""
    priors = (torch.tensor([0.0, 1.0, -0.4, 0.8]), torch.tensor([0.7, 0.3, 0.2, 0.4]))  # bg,RV,myo,cav
    cfg = SynthCfg(synth_p=1.0, realistic=True, jitter=0.0, deform=0.0, std_scale=(0.0, 0.0),
                   bias_strength=0.0, blur=(0.0, 0.0), noise=0.0)
    torch.manual_seed(0)
    img, _ = synthesize_from_labels(_mask(1), cfg, N, priors)
    # blood (RV, cav) must be brighter than myocardium in the generated image
    band = lambda c: img[0, 0, c * 2:(c + 1) * 2].mean()
    assert band(1) > band(2) and band(3) > band(2)     # RV>myo and cav>myo (blood bright)


def test_deform_invents_anatomy():
    m = _mask()
    torch.manual_seed(1)
    _, warped = synthesize_from_labels(m, SynthCfg(synth_p=1.0, deform=0.3), N)
    assert not torch.equal(warped, m)
    assert set(warped.unique().tolist()).issubset(set(range(N)))
    _, same = synthesize_from_labels(m, SynthCfg(synth_p=1.0, deform=0.0), N)
    assert torch.equal(same, m)


def test_seed_deterministic():
    cfg, m = SynthCfg(synth_p=1.0, realistic=False), _mask()
    torch.manual_seed(7); a, ma = synthesize_from_labels(m, cfg, N)
    torch.manual_seed(7); b, mb = synthesize_from_labels(m, cfg, N)
    assert torch.equal(a, b) and torch.equal(ma, mb)


def test_load_synth_priors_from_reference(tmp_path):
    """Priors round-trip through the reference store: a synth_priors.yaml -> (means,stds) tensors
    indexed by class (0=bg,1=RV,2=myo,3=cav); unverified entries are skipped (returns None)."""
    from core.reference import Reference
    from cardioseg.data.priors import load_synth_priors
    (tmp_path / "synth_priors.yaml").write_text(
        "synth_priors:\n  intensity:\n"
        "    background: {value: [0.0, 0.7], source: computed, based_on: t, extracted_by: computed, verified: true}\n"
        "    RV:      {value: [1.0, 0.3], source: computed, based_on: t, extracted_by: computed, verified: true}\n"
        "    LV-myo:  {value: [-0.3, 0.2], source: computed, based_on: t, extracted_by: computed, verified: true}\n"
        "    LV-cav:  {value: [0.8, 0.4], source: computed, based_on: t, extracted_by: computed, verified: true}\n")
    ref = Reference(root=tmp_path)
    means, stds = load_synth_priors(N, "cpu", ref=ref)
    assert torch.allclose(means, torch.tensor([0.0, 1.0, -0.3, 0.8]), atol=1e-5)
    assert torch.allclose(stds, torch.tensor([0.7, 0.3, 0.2, 0.4]), atol=1e-5)
    # blood (RV, cav) brighter than myo — the ordering that makes synth transferable
    assert means[1] > means[2] and means[3] > means[2]


def test_partition_bg_structures_image_keeps_target():
    """bg_mode='partition' splits the background by REAL per-slice intensity into tiers -> the image's
    bg gets structure (>1 level), but the returned TARGET keeps only real labels (bg stays 0). This is
    the fix that took pure-synth 0.39 -> 0.66."""
    Y = _mask()
    real = torch.randn(2, 1, 8, 8)                       # varied bg intensity -> tiers differ
    priors = (torch.tensor([0.0, 1.0, -0.3, 0.8]), torch.tensor([0.3, 0.2, 0.2, 0.3]))
    cfg = SynthCfg(synth_p=1.0, realistic=True, bg_mode="partition", bg_tiers=4, deform=0.0,
                   jitter=0.0, std_scale=(0.0, 0.0), bias_strength=0.0, blur=(0.0, 0.0), noise=0.0)
    torch.manual_seed(0)
    img, msk = synthesize_from_labels(Y, cfg, N, priors, real_img=real)
    assert set(msk.unique().tolist()).issubset(set(range(N)))     # target = real labels only (bg=0)
    bg = (Y[0] == 0)
    assert img[0, 0][bg].unique().numel() > 1                     # bg painted with structure, not flat


def test_pv_softens_boundaries():
    """pv_sigma>0 blurs the class-MEAN map -> boundary voxels are mixes (intermediate values) instead
    of hard per-class levels (the partial-volume physical upgrade)."""
    priors = (torch.tensor([0.0, 1.0, -0.3, 0.8]), torch.zeros(4))   # std 0 -> flat per class
    base = dict(synth_p=1.0, realistic=True, bg_mode="flat", deform=0.0, jitter=0.0,
                std_scale=(0.0, 0.0), bias_strength=0.0, blur=(0.0, 0.0), noise=0.0)
    torch.manual_seed(0)
    hard, _ = synthesize_from_labels(_mask(1), SynthCfg(pv_sigma=0.0, **base), N, priors)
    soft, _ = synthesize_from_labels(_mask(1), SynthCfg(pv_sigma=1.0, **base), N, priors)
    assert hard[0, 0].unique().numel() == N          # hard edges: one level per class
    assert soft[0, 0].unique().numel() > N           # PV mixing -> intermediate boundary values


def test_kspace_lowpass_changes_and_keeps_shape():
    """k-space truncation removes high frequencies -> output differs from no-PSF, same shape, finite."""
    off = SynthCfg(synth_p=1.0, realistic=False, bg_mode="flat", kspace=0.0)
    on = SynthCfg(synth_p=1.0, realistic=False, bg_mode="flat", kspace=0.5)
    torch.manual_seed(0); a, _ = synthesize_from_labels(_mask(), off, N)
    torch.manual_seed(0); b, _ = synthesize_from_labels(_mask(), on, N)
    assert b.shape == a.shape and torch.isfinite(b).all()
    assert not torch.allclose(a, b)
