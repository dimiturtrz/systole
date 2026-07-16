"""bd 753u: EarlyStop decision logic — raw-max best-tracking + min_delta noise filter."""
from cardioseg.training.earlystop import EarlyStop


def test_raw_max_improves_then_stops():
    es = EarlyStop(patience=2)                       # min_delta=0 -> plain raw-max
    assert es.update(0.5) is True and es.best == 0.5
    assert es.update(0.6) is True                    # new max
    assert es.update(0.55) is False and es.stop is False   # 1 bad
    assert es.update(0.59) is False and es.stop is True    # 2 bad -> patience hit


def test_min_delta_ignores_sub_threshold_gains():
    es = EarlyStop(patience=3, min_delta=0.01)
    assert es.update(0.50) is True
    assert es.update(0.505) is False                 # +0.005 < min_delta -> not an improvement
    assert es.update(0.515) is True                  # +0.015 > min_delta -> improvement (resets)


def test_best_tracks_the_max_not_the_last():
    es = EarlyStop(patience=5)
    for v in (0.40, 0.42, 0.90, 0.43):               # 0.90 = the peak
        es.update(v)
    assert es.best == 0.90                            # raw-max keeps the best-val weights' epoch
    assert es.bad == 1                                # one bad since the peak
