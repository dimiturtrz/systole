"""Nyúl–Udupa histogram standardization (harmonization, bd cardiac-seg-qfz).

The *strip* force for vendor intensity variance: MRI intensity is uncalibrated, so the same tissue maps
to different values across scanners. z-score fixes only mean/scale; Nyúl aligns the whole intensity
DISTRIBUTION by matching landmark percentiles to a learned standard scale — piecewise-linear so tissue
ordering is preserved. Two steps:
  1. FIT (once, on a cohort): each image's intensity at a set of percentiles, rescaled to a common
     range, averaged -> the STANDARD landmark scale. These landmarks are a normalization axis -> stored
     as reference data (provenance), not recomputed per run.
  2. TRANSFORM (per image): piecewise-linearly map the image's own percentiles onto the standard.

Pure numpy, unit-testable. Fit is data-derived (reference); transform is applied in preprocessing.
Ref: Nyúl, Udupa & Zhang, IEEE TMI 2000. Deep-dive 2026-06-21_intensity-normalization-and-harmonization.
"""
from __future__ import annotations

import numpy as np

# percentile landmarks: robust tails (p1/p99) + deciles. The tails anchor the scale, deciles the shape.
LANDMARKS: tuple[int, ...] = (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99)


class Nyul:
    """Nyúl histogram standardization (harmonization) bound to a fitted STANDARD landmark scale: construct
    once with the learned standard (the fit-side session), then `transform` many images onto it. The fit
    helpers (`image_landmarks`, `fit_standard`) stay static — they run BEFORE an instance exists, to derive
    the standard a `Nyul(standard)` is then constructed with."""

    def __init__(self, standard: np.ndarray):
        self.standard = np.asarray(standard, dtype=np.float64)   # fitted standard landmark scale (session)

    @staticmethod
    def image_landmarks(img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Intensity at each percentile in LANDMARKS, over `mask` (foreground) if given else the whole image."""
        v = img[mask.astype(bool)] if mask is not None else img.reshape(-1)
        return np.percentile(v.astype(np.float64), LANDMARKS)

    @staticmethod
    def fit_standard(landmark_rows: np.ndarray) -> np.ndarray:
        """Learn the standard scale from per-image landmarks [N, len(LANDMARKS)]: rescale each image's
        landmarks to [0,1] by its own (first,last) landmark, then average -> the standard landmark vector
        (monotonic, in [0,1]). Robust to each image's arbitrary intensity range."""
        rows = np.asarray(landmark_rows, dtype=np.float64)
        lo, hi = rows[:, :1], rows[:, -1:]
        scaled = (rows - lo) / np.clip(hi - lo, 1e-6, None)
        return scaled.mean(axis=0)

    def transform(self, img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Map `img` onto this instance's standard scale: piecewise-linear interp from the image's own
        landmarks to the standard landmarks. Values outside the landmark range extrapolate linearly at the
        end segments (np clamps to the standard endpoints — acceptable, tails are p1/p99). Returns a float
        array."""
        lm = Nyul.image_landmarks(img, mask)
        lm = Nyul._dedup_monotone(lm)                              # np.interp needs strictly-increasing xp
        return np.interp(img.astype(np.float64), lm, self.standard)

    @staticmethod
    def _dedup_monotone(lm: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Force strictly increasing landmarks (flat histograms can tie percentiles) so np.interp is valid."""
        out = lm.astype(np.float64).copy()
        for i in range(1, len(out)):
            if out[i] <= out[i - 1]:
                out[i] = out[i - 1] + eps
        return out
