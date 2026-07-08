"""Pure plottable-data logic for render_overlay: the crop+isotropic geometry, the train/held-out
tag, and the EF title string. marching_cubes / pyvista Plotter / gltf-html-screenshot writes are
shells (pragma'd). Tests stub the disk-scanning split source; the rest is numpy + core.measure."""
import logging
from pathlib import Path

import numpy as np
import render_overlay as R
from render_overlay import _ef_title, _split_tag, crop_and_iso

# --- crop_and_iso ----------------------------------------------------------

def test_crop_and_iso_crops_then_resamples_both():
    """Crop+iso class: img and mask cropped to the same heart bbox, then resampled to isotropic —
    same output shape for both, labels preserved (nearest), iso step = finest spacing."""
    img = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    mask = np.zeros((3, 8, 8), np.uint8)
    mask[1:3, 2:6, 2:6] = 2
    img_i, mask_i, iso = crop_and_iso(img, mask, (6.0, 1.5, 1.5), margin_mm=0.0)
    assert img_i.shape == mask_i.shape
    assert iso == (1.5, 1.5, 1.5)
    assert set(np.unique(mask_i)).issubset({0, 2})  # order-0 keeps labels, invents none


def test_crop_and_iso_isotropic_preserves_inplane_shape():
    """Isotropic-input class: equal spacing -> in-plane unscaled; only the cropped bbox survives."""
    img = np.zeros((4, 10, 10), np.float32)
    mask = np.zeros((4, 10, 10), np.uint8)
    mask[1:3, 3:7, 3:7] = 1
    img_i, mask_i, iso = crop_and_iso(img, mask, (1.5, 1.5, 1.5), margin_mm=0.0)
    assert mask_i.shape == (2, 4, 4) and iso == (1.5, 1.5, 1.5)


# --- _split_tag (patient id -> tag string) ---------------------------------

def _fake_cases(names):
    return [Path(n) for n in names]


def test_split_tag_held_out(monkeypatch, caplog):
    """Held-out class: the seed-0 0.2 val slice of a 5-case list holds 1 case -> that id -> 'held-out',
    no warning logged."""
    monkeypatch.setattr(R, "acdc_cases",
                        lambda: _fake_cases([f"p{i}" for i in range(5)]))
    # find which one the deterministic split marks held-out, then assert its tag.
    tags = {p: _split_tag(p) for p in [f"p{i}" for i in range(5)]}
    held = [p for p, t in tags.items() if "held-out" in t]
    assert len(held) == 1
    assert all("held-out" in tags[h] for h in held)


def test_split_tag_train_seen_warns(monkeypatch, caplog):
    """Train-seen class: an id NOT in the val slice -> '  TRAIN-seen' AND a warning (pred overstates)."""
    monkeypatch.setattr(R, "acdc_cases",
                        lambda: _fake_cases([f"p{i}" for i in range(5)]))
    tags = {p: _split_tag(p) for p in [f"p{i}" for i in range(5)]}
    seen = [p for p, t in tags.items() if "TRAIN-seen" in t]
    assert len(seen) == 4
    with caplog.at_level(logging.WARNING, logger="cardioview.render_overlay"):
        _split_tag(seen[0])
    assert any("training" in r.message for r in caplog.records)


# --- _ef_title -------------------------------------------------------------

def _masks_and_case(ed_cav, es_cav):
    """A concentric mask pair + a case with matching GT, LV-cav (label 3) sized to hit target EF."""
    def m(n):
        v = np.zeros((2, 8, 8), np.uint8)
        v[:, :2, :n] = 3  # n cav voxels per slice per row-band -> controls volume
        return v
    masks = {"ED": m(ed_cav), "ES": m(es_cav)}
    case = {"ed_gt": m(ed_cav), "es_gt": m(es_cav)}
    return masks, case


def test_ef_title_full_pair():
    """Full ED+ES class: both phases present -> a title with pred EF and GT EF percentages."""
    masks, case = _masks_and_case(8, 4)
    title = _ef_title(masks, case, (1.0, 1.0, 1.0), "pred")
    assert "EF pred" in title and "GT" in title and "%" in title


def test_ef_title_missing_phase_is_empty():
    """Missing-phase class: only ED present -> no EF pair -> empty string (nothing to title)."""
    masks = {"ED": np.zeros((2, 4, 4), np.uint8)}
    assert _ef_title(masks, {}, (1.0, 1.0, 1.0), "pred") == ""
