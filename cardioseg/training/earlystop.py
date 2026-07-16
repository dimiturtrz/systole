"""Early-stopping decision, isolated from the GPU train loop so it's unit-testable (bd 753u).

Raw-max on the val-Dice: snapshot the best-val weights, stop after `patience` epochs with no gain.
`min_delta` is the smallest gain that counts (filters numerical noise; the structural ±0.05 val noise is
far above it, so this is a minor guard, not a speed lever). EMA-smoothing the stop signal was TRIED and
KILLED (bd 753u): it decoupled the snapshot from the actual best-val epoch and stopped later — test −0.052.
"""


class EarlyStop:
    """Stateful raw-max early stop. `update(metric)` returns whether this step is a new best (the caller
    snapshots weights then); `.stop` is True once `patience` steps pass with no new best."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience, self.min_delta = patience, min_delta
        self.best = -1.0
        self.bad = 0

    def update(self, metric: float) -> bool:
        """Return True on a new best (resets the patience counter); else count a bad epoch."""
        if metric > self.best + self.min_delta:
            self.best, self.bad = metric, 0
            return True
        self.bad += 1
        return False

    @property
    def stop(self) -> bool:
        return self.bad >= self.patience
