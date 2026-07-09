"""core.obs tests: logger setup, the progress wrapper, and the timed context manager.

The pure/observable behaviour is pinned — setup configures the named logger (level, no root
propagation, optional file handler that truncates), progress wraps an iterable losslessly, and
timed logs START/DONE around its block. tqdm/logging internals are not re-tested.
"""
import logging

from core.obs import Obs, timed


def test_setup_configures_named_logger():
    """setup returns the 'cardioseg' logger with the level set + propagate off (survives a lib's
    logging.basicConfig(force=True) wiping root)."""
    log = Obs.setup(level=logging.DEBUG)
    assert log.name == "cardioseg"
    assert log.level == logging.DEBUG
    assert log.propagate is False
    assert any(isinstance(h, logging.StreamHandler) for h in log.handlers)


def test_setup_with_file_truncates_and_adds_handler(tmp_path):
    """A logfile -> its parent is created, the file is truncated at start, a file handler is added,
    and an emitted record lands on disk."""
    logfile = tmp_path / "sub" / "train.log"
    logfile.parent.mkdir()
    logfile.write_text("stale")                      # pre-existing content must be truncated
    log = Obs.setup(logfile)
    assert logfile.read_text() == ""                 # truncated at setup
    log.info("hello")
    assert "hello" in logfile.read_text()


def test_progress_wraps_iterable_losslessly():
    """progress yields exactly the underlying items (degrades to a plain pass-through in non-tty)."""
    items = [1, 2, 3]
    assert list(Obs.progress(items, "desc", total=len(items))) == items


def test_timed_logs_start_and_done(caplog):
    """timed logs START on enter and DONE (+elapsed) on exit, on the given logger."""
    log = logging.getLogger("cardioseg.test_timed")
    with caplog.at_level(logging.INFO, logger=log.name):
        with timed(log, "load data"):
            pass
    msgs = [r.getMessage() for r in caplog.records]
    assert any("START load data" in m for m in msgs)
    assert any("DONE" in m and "load data" in m for m in msgs)
