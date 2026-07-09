"""Store normalization stage (core.data.static.store.normalize) — equivalence classes over the recipe-
bound per-case transform: ref_path composition, load_standard (present -> ndarray / absent -> None),
and apply_case wiring (delegates to Preprocess.preprocess_case, threading the recipe fields; nyul_standard
gated on the nyul flag). fit_standard is pragma-no-cover disk I/O — a light construction check only. Fed
synthetic inputs / a monkeypatched Preprocess + Reference — I/O-free."""
import numpy as np

from core.data.static.reference import Reference
from core.data.static.store import normalize as normalize_mod
from core.data.static.store.normalize import Normalizer
from core.preprocessing.n4 import N4Cfg


# --- ref_path: reference-dir composition ---
def test_ref_path_under_reference_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CARDIAC_DATA", str(tmp_path))
    p = Normalizer.ref_path()
    assert p.name == "nyul.yaml"                                  # the fitted-standard sidecar name
    assert p.parent == Reference.reference_dir()                 # under the reference dir


# --- load_standard: present -> float ndarray / absent -> None ---
def test_load_standard_absent_is_none(monkeypatch):
    """No 'nyul'/'standard' entry -> None (then nyul can't run)."""
    monkeypatch.setattr(Reference, "get", lambda self, *keys: None)
    assert Normalizer.load_standard() is None


def test_load_standard_present_is_ndarray(monkeypatch):
    """A stored standard -> float64 ndarray of the landmark values."""
    monkeypatch.setattr(Reference, "get", lambda self, *keys: [0.0, 0.5, 1.0])
    std = Normalizer.load_standard()
    assert isinstance(std, np.ndarray) and std.dtype == np.float64
    assert std.tolist() == [0.0, 0.5, 1.0]


# --- apply_case: delegates to Preprocess.preprocess_case, threading the recipe ---
def _capture_preprocess(monkeypatch):
    captured = {}

    def _fake(case, **kw):
        captured.update(case=case, **kw)
        return {"ed_img": np.zeros((1, 1))}

    monkeypatch.setattr(normalize_mod.Preprocess, "preprocess_case", staticmethod(_fake))
    return captured


def test_apply_case_threads_recipe(tmp_path, monkeypatch):
    """apply_case forwards inplane/n4/n4_params/norm and the loader to preprocess_case."""
    captured = _capture_preprocess(monkeypatch)
    cfg = N4Cfg()
    n = Normalizer(1.25, n4=True, n4_params=cfg, nyul=False, norm="blood")
    loader = object()
    out = n.apply_case(tmp_path / "s1", loader)
    assert out == {"ed_img": np.zeros((1, 1))} or "ed_img" in out
    assert captured["target_inplane"] == 1.25 and captured["n4"] is True
    assert captured["n4_params"] is cfg and captured["norm"] == "blood"
    assert captured["loader"] is loader and captured["case"] == tmp_path / "s1"


def test_apply_case_nyul_standard_gated_on_flag(tmp_path, monkeypatch):
    """nyul=False -> nyul_standard passed as None (even if a standard is held); nyul=True -> the standard."""
    std = np.array([0.0, 1.0])

    captured = _capture_preprocess(monkeypatch)
    Normalizer(nyul=False, nyul_standard=std).apply_case(tmp_path / "s", None)
    assert captured["nyul_standard"] is None                     # gate off -> None

    captured = _capture_preprocess(monkeypatch)
    Normalizer(nyul=True, nyul_standard=std).apply_case(tmp_path / "s", None)
    assert captured["nyul_standard"] is std                      # gate on -> the standard


# --- construction: recipe fields stored verbatim (defaults + explicit) ---
def test_construction_defaults_and_explicit():
    d = Normalizer()
    assert d.n4 is False and d.nyul is False and d.norm == "zscore" and d.nyul_standard is None
    e = Normalizer(2.0, n4=True, nyul=True, norm="blood")
    assert e.inplane == 2.0 and e.n4 is True and e.nyul is True and e.norm == "blood"
