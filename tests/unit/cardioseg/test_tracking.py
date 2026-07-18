"""Tracking is optional + guarded — disabled -> no-op handle that never raises; config flattens."""
from typing import Any, ClassVar

import cardioseg.tracking as trk
from cardioseg.tracking import Tracker, _Live, _Noop

_flat = Tracker._flat


def start(experiment, run_name, params=None, tags=None):
    """Open a fresh run via the session ctor + instance start() (was the Tracker.start factory)."""
    return Tracker(experiment, run_name, params, tags).start()


def track_run(experiment, run_name, run_dir=None, params=None, tags=None):
    """Resume-or-create via the session ctor + instance track_run() (was the factory)."""
    return Tracker(experiment, run_name, params, tags).track_run(run_dir=run_dir)


def test_flat_nested():
    assert _flat({"a": 1, "b": {"c": 2, "d": {"e": 3}}}) == {"a": 1, "b.c": 2, "b.d.e": 3}


def test_flat_flat_and_empty():
    """Boundary: already-flat dict is unchanged; empty -> empty."""
    assert _flat({"lr": 1e-3}) == {"lr": 1e-3}
    assert _flat({}) == {}


def test_noop_when_disabled(monkeypatch):
    """CARDIOSEG_NO_MLFLOW -> no-op handle; all calls are safe (training must never depend on it)."""
    monkeypatch.setenv("CARDIOSEG_NO_MLFLOW", "1")
    h = start("cardioseg", "run", {"model": {"channels": 16}, "lr": 1e-3})
    assert isinstance(h, _Noop)
    h.metric("val_dice", 0.9, step=0)
    h.summary({"test": {"dice_mean": 0.84, "ef_mae": 11.0}})
    h.artifact("does/not/exist")
    h.tag("k", "v")
    h.log_model(None, "m")
    h.end()                                    # none of these raise


class _FakeMlflow:
    """Records calls so _Live's guarded delegation is observable without a real mlflow backend."""
    def __init__(self):
        self.metrics, self.tags, self.artifacts = [], [], []

    def log_metric(self, key, value, step=None):
        self.metrics.append((key, value, step))

    def set_tag(self, key, value):
        self.tags.append((key, value))

    def log_artifact(self, path):
        self.artifacts.append(path)

    def end_run(self):
        self.tags.append(("__ended__", True))


def test_live_metric_and_tag():
    """_Live class: metric coerces to float; tag stringifies — both delegate to mlflow."""
    fake = _FakeMlflow()
    live = _Live(fake)
    live.metric("val_dice", 0.9, step=3)
    live.tag("phase", 2)
    assert fake.metrics == [("val_dice", 0.9, 3)]
    assert fake.tags == [("phase", "2")]


def test_live_summary_logs_only_known_scalar_axes():
    """_Live.summary class: only dict axes with numeric dice_mean/ef_mae become fit_* metrics."""
    fake = _FakeMlflow()
    _Live(fake).summary({
        "val": {"dice_mean": 0.84, "ef_mae": 11.0, "other": "x"},
        "skip": {"dice_mean": None},          # non-numeric -> ignored
        "bad": "not-a-dict",                  # non-dict axis -> ignored
    })
    keys = {k for k, _, _ in fake.metrics}
    assert keys == {"fit_val_dice_mean", "fit_val_ef_mae"}


def test_live_artifact_only_existing(tmp_path):
    """_Live.artifact class: logs an existing path, silently skips a missing one."""
    fake = _FakeMlflow()
    live = _Live(fake)
    f = tmp_path / "m.json"; f.write_text("{}")
    live.artifact(f)
    live.artifact(tmp_path / "nope.json")
    assert fake.artifacts == [str(f)]


def test_live_end_guarded():
    """_Live.end class: delegates end_run, swallows nothing needed here."""
    fake = _FakeMlflow()
    _Live(fake).end()
    assert ("__ended__", True) in fake.tags


class _RaisingMlflow:
    """Every logging call raises — exercises _Live's guard branches (must never propagate)."""
    def log_metric(self, *a, **k): raise RuntimeError("boom")
    def log_artifact(self, *a, **k): raise RuntimeError("boom")
    def set_tag(self, *a, **k): raise RuntimeError("boom")
    def end_run(self, *a, **k): raise RuntimeError("boom")


def test_live_methods_swallow_backend_errors(tmp_path):
    """_Live boundary: a raising backend never propagates (metric/artifact/tag/end all guarded)."""
    live = _Live(_RaisingMlflow())
    f = tmp_path / "x.json"; f.write_text("{}")
    live.metric("d", 1.0)          # log_metric raises -> swallowed
    live.artifact(f)               # log_artifact raises -> swallowed
    live.tag("k", "v")             # set_tag raises -> swallowed
    live.end()                     # end_run raises -> swallowed


class _LogModelMlflow:
    """Fake with the log_model surface: mlflow.pytorch.log_model + a client returning versions."""
    class pytorch:
        logged: ClassVar[list[Any]] = []
        @classmethod
        def log_model(cls, model, name=None, registered_model_name=None):
            cls.logged.append((model, registered_model_name))

    class _Client:
        def __init__(self, outer): self.outer = outer
        def search_model_versions(self, _filter):
            return [type("MV", (), {"version": "2"})(), type("MV", (), {"version": "5"})()]
        def update_model_version(self, name, ver, description=None):
            self.outer.descs.append((ver, description))
        def set_model_version_tag(self, name, ver, k, v):
            self.outer.vtags.append((ver, k, v))
        def set_registered_model_alias(self, name, alias, ver):
            self.outer.aliases.append((alias, ver))

    def __init__(self):
        self.descs, self.vtags, self.aliases = [], [], []
        self.tracking = type(
            "T", (), {"MlflowClient": staticmethod(lambda outer=self: _LogModelMlflow._Client(outer))},
        )()


def test_live_log_model_registers_version(monkeypatch):
    """_Live.log_model class: logs model, resolves max version, applies description/tags/alias."""
    fake = _LogModelMlflow()
    fake.pytorch.logged = []
    monkeypatch.setattr(trk, "mlflow", fake)              # module-level mlflow.pytorch.log_model
    live = _Live(fake)
    live.log_model("MODEL", "cardioseg-2dunet", alias="production",
                   description="flagship", version_tags={"dice": 0.9})
    assert fake.pytorch.logged == [("MODEL", "cardioseg-2dunet")]
    assert ("5", "flagship") in fake.descs                # max(2,5)=5
    assert ("5", "dice", "0.9") in fake.vtags
    assert ("production", "5") in fake.aliases


def test_live_log_model_swallows_errors(monkeypatch):
    """_Live.log_model boundary: a raising backend is fully guarded (never propagates)."""
    class _Boom:
        class pytorch:
            @staticmethod
            def log_model(*a, **k): raise RuntimeError("boom")
    monkeypatch.setattr(trk, "mlflow", _Boom)
    _Live(_Boom()).log_model("m", "name")                 # no raise


def test_start_disabled_returns_noop(monkeypatch):
    """start boundary: opt-out env -> _Noop (no mlflow touched)."""
    monkeypatch.setenv("CARDIOSEG_NO_MLFLOW", "1")
    assert isinstance(start("exp", "run"), _Noop)


def test_track_run_disabled_returns_noop(monkeypatch):
    """track_run boundary: opt-out env -> _Noop."""
    monkeypatch.setenv("CARDIOSEG_NO_MLFLOW", "1")
    assert isinstance(track_run("exp", "run"), _Noop)


class _RunInfo:
    def __init__(self, rid): self.info = type("I", (), {"run_id": rid})()


class _StartFakeMlflow(_FakeMlflow):
    """Fake with the module-level surface start()/track_run() call (no sqlite/network)."""
    def __init__(self, rid="RID123"):
        super().__init__()
        self._rid = rid
        self.params, self.started = {}, []

    def set_tracking_uri(self, *_): pass
    def set_experiment(self, *_): pass
    def enable_system_metrics_logging(self): pass

    def start_run(self, run_name=None, run_id=None):
        self.started.append(run_id or run_name)

    def log_params(self, p): self.params.update(p)
    def active_run(self): return _RunInfo(self._rid)


def test_start_live_logs_params_and_tags(monkeypatch, tmp_path):
    """start class (enabled): flattens+logs params, sets tags, returns _Live."""
    monkeypatch.delenv("CARDIOSEG_NO_MLFLOW", raising=False)
    fake = _StartFakeMlflow()
    monkeypatch.setattr(trk, "mlflow", fake)
    monkeypatch.setattr(trk, "_MLRUNS", tmp_path / "mlruns")
    h = start("exp", "run1", {"model": {"channels": 8}}, tags={"stage": "dev"})
    assert isinstance(h, _Live)
    assert fake.params == {"model.channels": 8}
    assert ("stage", "dev") in fake.tags


def test_track_run_fresh_persists_id(monkeypatch, tmp_path):
    """track_run class: no id-file -> fresh run, writes .mlflow_run_id for later resume."""
    monkeypatch.delenv("CARDIOSEG_NO_MLFLOW", raising=False)
    fake = _StartFakeMlflow(rid="NEWID")
    monkeypatch.setattr(trk, "mlflow", fake)
    monkeypatch.setattr(trk, "_MLRUNS", tmp_path / "mlruns")
    run_dir = tmp_path / "runs" / "foo"; run_dir.mkdir(parents=True)
    h = track_run("exp", "run", run_dir=run_dir, params={"lr": 1e-3})
    assert isinstance(h, _Live)
    assert (run_dir / ".mlflow_run_id").read_text() == "NEWID"


def test_track_run_resumes_existing_id(monkeypatch, tmp_path):
    """track_run boundary: existing id-file -> resume by run_id, don't re-log params."""
    monkeypatch.delenv("CARDIOSEG_NO_MLFLOW", raising=False)
    fake = _StartFakeMlflow()
    monkeypatch.setattr(trk, "mlflow", fake)
    monkeypatch.setattr(trk, "_MLRUNS", tmp_path / "mlruns")
    run_dir = tmp_path / "runs" / "bar"; run_dir.mkdir(parents=True)
    (run_dir / ".mlflow_run_id").write_text("OLDID")
    track_run("exp", "run", run_dir=run_dir, params={"lr": 1e-3})
    assert fake.started == ["OLDID"] and fake.params == {}   # resumed, no params relogged


def test_track_run_fresh_sets_tags(monkeypatch, tmp_path):
    """track_run class: fresh run with tags -> tags applied (the tag loop) + _Live returned."""
    monkeypatch.delenv("CARDIOSEG_NO_MLFLOW", raising=False)
    fake = _StartFakeMlflow(rid="TID")
    monkeypatch.setattr(trk, "mlflow", fake)
    monkeypatch.setattr(trk, "_MLRUNS", tmp_path / "mlruns")
    run_dir = tmp_path / "runs" / "baz"; run_dir.mkdir(parents=True)
    track_run("exp", "run", run_dir=run_dir, tags={"stage": "eval"})
    assert ("stage", "eval") in fake.tags


class _SysMetricsRaises(_StartFakeMlflow):
    """enable_system_metrics_logging raises (psutil/pynvml absent) — the inner guard must swallow it."""
    def enable_system_metrics_logging(self): raise RuntimeError("no pynvml")


def test_start_system_metrics_failure_is_guarded(monkeypatch, tmp_path):
    """start boundary: system-metrics logging raising doesn't abort the run -> still _Live."""
    monkeypatch.delenv("CARDIOSEG_NO_MLFLOW", raising=False)
    monkeypatch.setattr(trk, "mlflow", _SysMetricsRaises())
    monkeypatch.setattr(trk, "_MLRUNS", tmp_path / "mlruns")
    assert isinstance(start("exp", "run"), _Live)


def test_track_run_system_metrics_failure_is_guarded(monkeypatch, tmp_path):
    """track_run boundary: system-metrics logging raising is swallowed -> still _Live."""
    monkeypatch.delenv("CARDIOSEG_NO_MLFLOW", raising=False)
    monkeypatch.setattr(trk, "mlflow", _SysMetricsRaises(rid="SID"))
    monkeypatch.setattr(trk, "_MLRUNS", tmp_path / "mlruns")
    run_dir = tmp_path / "runs" / "sm"; run_dir.mkdir(parents=True)
    assert isinstance(track_run("exp", "run", run_dir=run_dir), _Live)


class _SetupRaises(_StartFakeMlflow):
    """set_experiment raises — the OUTER guard converts any setup failure to _Noop."""
    def set_experiment(self, *_): raise RuntimeError("sqlite locked")


def test_start_setup_failure_returns_noop(monkeypatch, tmp_path):
    """start boundary: a fatal setup error -> _Noop (tracking must never break a run)."""
    monkeypatch.delenv("CARDIOSEG_NO_MLFLOW", raising=False)
    monkeypatch.setattr(trk, "mlflow", _SetupRaises())
    monkeypatch.setattr(trk, "_MLRUNS", tmp_path / "mlruns")
    assert isinstance(start("exp", "run"), _Noop)


def test_track_run_setup_failure_returns_noop(monkeypatch, tmp_path):
    """track_run boundary: a fatal setup error -> _Noop."""
    monkeypatch.delenv("CARDIOSEG_NO_MLFLOW", raising=False)
    monkeypatch.setattr(trk, "mlflow", _SetupRaises())
    monkeypatch.setattr(trk, "_MLRUNS", tmp_path / "mlruns")
    assert isinstance(track_run("exp", "run"), _Noop)
