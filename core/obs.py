"""Observability: file+console logging and a timing context manager, for hunting bottlenecks.

Subprocess wrappers (uv run, conda run, CI capture) often buffer child stdout, so progress printed
to stdout is invisible until exit. Logging through a FileHandler writes straight from the process to
disk — tail the file to watch live.

    from core.obs import Obs, timed
    log = Obs.setup("runs/foo/train.log")
    with timed(log, "load data"):
        ...
    for x in Obs.progress(loader, "epoch 0", total=len(loader)):
        ...
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from tqdm import tqdm


class _AppendHandler(logging.Handler):
    """Opens the file, writes the line, closes — every emit. Immune to logging.shutdown() closing a
    long-lived handle (MONAI/torch call it mid-run, which silently kills a normal FileHandler). Fine
    here: logging is per-phase/per-epoch (low frequency), so re-open cost is negligible."""

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = str(path)

    def emit(self, record):
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(self.format(record) + "\n")
        except OSError:
            self.handleError(record)


class Obs:
    """Observability entrypoints (the free helpers folded in as staticmethods, public names kept):
    logger `setup` + a `progress` bar. The `timed` context manager stays its own class."""

    @staticmethod
    def setup(logfile: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
        """Configure the `cardioseg` logger -> console (stdout) + optional file. Returns it.

        Handlers go on the NAMED logger with propagate=False (not the root) — third-party libs
        (numexpr/MONAI/torch) call logging.basicConfig(force=True) which wipes root handlers; keeping
        ours off the root makes them survive that. cardioseg.* children propagate up to here.
        """
        log = logging.getLogger("cardioseg")
        log.setLevel(level)
        log.propagate = False
        log.handlers.clear()
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); log.addHandler(sh)
        if logfile is not None:
            Path(logfile).parent.mkdir(parents=True, exist_ok=True)
            Path(logfile).write_text("")                      # truncate at start
            fh = _AppendHandler(logfile); fh.setFormatter(fmt); log.addHandler(fh)
        return log

    @staticmethod
    def progress(iterable, desc: str, total: int | None = None, every: float = 5.0):
        """tqdm progress bar (degrades gracefully in non-tty).
        `every` = min seconds between bar refreshes so file logs stay readable."""
        return tqdm(iterable, desc=desc, total=total, mininterval=every, dynamic_ncols=True)


class timed:
    """Context manager that logs START/DONE + elapsed seconds — the basic bottleneck probe."""

    def __init__(self, log: logging.Logger, msg: str):
        self.log, self.msg = log, msg

    def __enter__(self):
        self.t = time.perf_counter()
        self.log.info("START %s", self.msg)
        return self

    def __exit__(self, *_):
        self.log.info("DONE  %s (%.2fs)", self.msg, time.perf_counter() - self.t)
