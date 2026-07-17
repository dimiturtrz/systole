"""Directed-generation envelope fit (core.data.analysis.directed). The testable cores are radial_psd —
the whole-FOV radial power spectrum used as the fit target — and psd_distance, the spectral-shape metric.
The grid fit + store loading need the painter/data and are exercised by the CLI, not here."""
import torch

from core.data.analysis.directed import Directed


def test_radial_psd_normalized_and_shaped():
    """Profile sums to 1 (a spectral shape, not a scale) and has n_bins-1 entries (DC dropped)."""
    torch.manual_seed(0)
    p = Directed.radial_psd(torch.randn(8, 1, 64, 64), n_bins=48)
    assert p.shape == (47,)
    assert abs(float(p.sum()) - 1.0) < 1e-5


def test_psd_distance_zero_for_identical():
    torch.manual_seed(0)
    p = Directed.radial_psd(torch.randn(8, 1, 64, 64))
    assert Directed.psd_distance(p, p) == 0.0


def test_psd_distance_symmetric():
    torch.manual_seed(0)
    a = Directed.radial_psd(torch.randn(8, 1, 64, 64))
    b = Directed.radial_psd(torch.randn(8, 1, 64, 64))
    assert abs(Directed.psd_distance(a, b) - Directed.psd_distance(b, a)) < 1e-9


def test_psd_distance_separates_white_from_lowpass():
    """A low-pass (blurred) field concentrates power at low k; white noise is flat -> the spectra differ,
    while white-vs-white is near zero. The metric's discriminating case (the xmcf plateau vs roll-off)."""
    torch.manual_seed(0)
    white = torch.randn(16, 1, 64, 64)
    lowpass = torch.nn.functional.avg_pool2d(white, 5, 1, 2)         # crude low-pass
    p_white, p_white2 = Directed.radial_psd(white), Directed.radial_psd(torch.randn(16, 1, 64, 64))
    p_low = Directed.radial_psd(lowpass)
    assert Directed.psd_distance(p_white, p_low) > 10 * Directed.psd_distance(p_white, p_white2)


def test_radial_psd_lowpass_shifts_mass_to_low_k():
    """Low-pass filtering moves spectral mass toward low frequencies (first bins gain, last bins lose)."""
    torch.manual_seed(0)
    white = torch.randn(16, 1, 64, 64)
    lowpass = torch.nn.functional.avg_pool2d(white, 5, 1, 2)
    p_white, p_low = Directed.radial_psd(white), Directed.radial_psd(lowpass)
    assert float(p_low[:5].sum()) > float(p_white[:5].sum())          # low-k mass up
    assert float(p_low[-5:].sum()) < float(p_white[-5:].sum())        # high-k mass down
