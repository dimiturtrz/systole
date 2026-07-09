"""Overlay hero-figure logic — the case-assembly + row-selection cores, decoupled from the model
forward + savefig shell. predict_volume runs on CPU (torch), so `_case` is tested with a stubbed
disk-load + prediction; `pick_hero_cases` is pure selection.
"""
import numpy as np

import cardioseg.evaluation.overlay as ov


def _lv_blob(d, h, w, r):
    m = np.zeros((d, h, w), np.uint8); m[1, h // 2 - r:h // 2 + r, w // 2 - r:w // 2 + r] = 3   # LV-cav
    return m


def test_case_assembles_row_with_matching_ef(monkeypatch):
    """Assembly class: given a loaded case + prediction (stubbed pred==gt), _case returns the
    mid-slice panel dict; EF_pred == EF_gt since the stub predicts the GT exactly."""
    ed, es = _lv_blob(3, 16, 16, 4), _lv_blob(3, 16, 16, 2)          # diastole larger than systole
    case = {"ed_img": ed.astype(np.float32), "es_img": es.astype(np.float32),
            "ed_gt": ed, "es_gt": es, "spacing": (10.0, 1.5, 1.5), "group": "NOR"}
    monkeypatch.setattr(ov, "load_arrays", lambda _p: case)
    monkeypatch.setattr(ov, "predict_volume",
                        lambda _m, img, _size, _dev, tta=True: (img > 0).astype(np.uint8) * 3)
    out = ov.Overlay._case(None, "/p/pt001.npz", size=16, device="cpu")
    assert out["group"] == "NOR" and out["name"] == "pt001"
    assert abs(out["ef_pred"] - out["ef_gt"]) < 1e-6                 # pred==gt -> equal EF
    assert out["img"].shape == (16, 16) and out["pred"].shape == (16, 16)


def test_pick_hero_cases_clean_min_err_and_worst_hcm():
    """Selection class: the clean row = lowest EF-error among DCM/NOR/MINF; the HCM row = WORST HCM."""
    cases = [{"group": "NOR", "ef_gt": 60.0, "ef_pred": 58.0},      # clean err 2
             {"group": "DCM", "ef_gt": 30.0, "ef_pred": 20.0},      # clean err 10
             {"group": "HCM", "ef_gt": 70.0, "ef_pred": 55.0},      # hcm err 15
             {"group": "HCM", "ef_gt": 65.0, "ef_pred": 62.0}]      # hcm err 3
    clean, hcm = ov.Overlay.pick_hero_cases(cases)
    assert clean["group"] == "NOR" and clean["ef_err"] == 2.0       # lowest clean error
    assert hcm["ef_err"] == 15.0                                    # worst HCM, not the best
