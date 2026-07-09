"""Registry ref-resolution + param flattening. The mlflow client/network is faked (no sqlite,
no artifact download) so only the pure branching logic (dir passthrough, alias->version->run-id
fallthrough, digit detection) is exercised."""
from mlflow.exceptions import MlflowException

import core.registry as reg
from core.registry import _flat, resolve


def test_flat_nested():
    """_flat class: nested dict -> dotted scalar keys."""
    assert _flat({"a": 1, "b": {"c": 2}}) == {"a": 1, "b.c": 2}


def test_mlflow_sets_tracking_uri_and_returns_module(monkeypatch):
    """_mlflow class: sets the sqlite tracking URI on the mlflow module, returns it."""
    seen = {}
    monkeypatch.setattr(reg.mlflow, "set_tracking_uri", lambda uri: seen.setdefault("uri", uri))
    m = reg._mlflow()
    assert m is reg.mlflow
    assert seen["uri"] == reg._DB_URI


def test_client_builds_from_mlflow(monkeypatch):
    """_client class: constructs an MlflowClient off the (URI-set) mlflow module."""
    sentinel = object()
    fake_tracking = type("T", (), {"MlflowClient": staticmethod(lambda: sentinel)})()
    monkeypatch.setattr(reg, "_mlflow", lambda: type("M", (), {"tracking": fake_tracking})())
    assert reg._client() is sentinel


def test_resolve_existing_dir_passthrough(tmp_path):
    """resolve class: a dir that already holds model.pth is returned as-is (no mlflow)."""
    (tmp_path / "model.pth").write_bytes(b"x")
    assert resolve(tmp_path) == tmp_path


class _MV:
    def __init__(self, rid): self.run_id = rid


class _FakeClient:
    """Fake MlflowClient: alias present only for 'production'."""
    def __init__(self):
        self.aliases = {"production": _MV("RID_PROD")}
        self.versions = {"7": _MV("RID_V7")}

    def get_model_version_by_alias(self, _name, ref):
        if ref in self.aliases:
            return self.aliases[ref]
        raise MlflowException("no such alias")

    def get_model_version(self, _name, ver):
        return self.versions[ver]


def test_run_id_for_alias(monkeypatch):
    """_run_id_for class: a known alias resolves via get_model_version_by_alias."""
    monkeypatch.setattr(reg, "_client", _FakeClient)
    assert reg._run_id_for("production") == "RID_PROD"


def test_run_id_for_version_number(monkeypatch):
    """_run_id_for class: alias miss + digit ref -> get_model_version."""
    monkeypatch.setattr(reg, "_client", _FakeClient)
    assert reg._run_id_for("7") == "RID_V7"


def test_run_id_for_raw_run_id(monkeypatch):
    """_run_id_for boundary: alias miss + non-digit -> assumed to already be a run-id."""
    monkeypatch.setattr(reg, "_client", _FakeClient)
    assert reg._run_id_for("abc123def") == "abc123def"


def test_resolve_downloads_and_flattens(monkeypatch, tmp_path):
    """resolve class: non-dir ref -> resolve id, download once, flatten model/ subdir if present."""
    monkeypatch.setattr(reg, "_client", _FakeClient)
    monkeypatch.setattr(reg, "_CACHE", tmp_path)
    monkeypatch.setattr(reg, "_mlflow", lambda: None)

    calls = []

    def _fake_download(*, run_id, artifact_path, dst_path):
        calls.append(run_id)
        inner = tmp_path / run_id / "model"
        inner.mkdir(parents=True, exist_ok=True)
        (inner / "model.pth").write_bytes(b"w")

    monkeypatch.setattr(reg.mlflow.artifacts, "download_artifacts", _fake_download)
    out = resolve("production")
    assert out == tmp_path / "RID_PROD" / "model"      # flattened to inner model/ dir
    assert calls == ["RID_PROD"]


def test_resolve_flattens_to_dst_when_no_inner(monkeypatch, tmp_path):
    """resolve boundary: artifacts land directly under dst (no model/ subdir) -> return dst."""
    monkeypatch.setattr(reg, "_client", _FakeClient)
    monkeypatch.setattr(reg, "_CACHE", tmp_path)
    monkeypatch.setattr(reg, "_mlflow", lambda: None)

    def _fake_download(*, run_id, artifact_path, dst_path):
        (tmp_path / run_id).mkdir(parents=True, exist_ok=True)
        (tmp_path / run_id / "model.pth").write_bytes(b"w")

    monkeypatch.setattr(reg.mlflow.artifacts, "download_artifacts", _fake_download)
    out = resolve("production")
    assert out == tmp_path / "RID_PROD"


class _MVer:
    def __init__(self, v): self.version = v


class _SaveClient:
    def __init__(self):
        self.artifacts, self.versions, self.tags, self.aliases, self.descs = [], [], [], [], []

    def log_artifact(self, rid, path, artifact_path=None):
        self.artifacts.append((rid, path))

    def create_model_version(self, name, source=None, run_id=None):
        self.versions.append((name, source, run_id)); return _MVer("5")

    def update_model_version(self, name, ver, description=None):
        self.descs.append((ver, description))

    def set_model_version_tag(self, name, ver, k, v):
        self.tags.append((ver, k, v))

    def set_registered_model_alias(self, name, alias, ver):
        self.aliases.append((alias, ver))


class _SaveMlflow:
    def set_tracking_uri(self, *_): pass


def test_save_model_reuses_run_id(monkeypatch, tmp_path):
    """save_model class: given run_id -> logs each staged file via client, registers a version,
    applies description/tags/alias. No own start/end run."""
    staging = tmp_path / "stage"; staging.mkdir()
    (staging / "model.pth").write_bytes(b"w")
    (staging / "config.json").write_text("{}")
    (staging / "sub").mkdir()                           # dir entry skipped (only files logged)

    client = _SaveClient()
    monkeypatch.setattr(reg, "_mlflow", _SaveMlflow)
    monkeypatch.setattr(reg, "_client", lambda: client)

    rid = reg.save_model(staging, run_name="r", run_id="RID9",
                         alias="production", description="flagship", tags={"dice": 0.9})
    assert rid == "RID9"
    assert len(client.artifacts) == 2                   # 2 files, dir skipped
    assert client.versions[0][2] == "RID9"
    assert ("5", "flagship") in client.descs
    assert ("5", "dice", "0.9") in client.tags
    assert ("production", "5") in client.aliases


class _OwnRunMlflow(_SaveMlflow):
    """Fake for the own-run branch (run_id=None): starts/ends its own run + logs artifacts itself."""
    def __init__(self):
        self.started = self.ended = False
        self.params, self.artifacts = {}, []

    def start_run(self, run_name=None): self.started = True
    def log_params(self, p): self.params.update(p)
    def active_run(self): return type("R", (), {"info": type("I", (), {"run_id": "AUTOID"})()})()
    def log_artifact(self, path, artifact_path=None): self.artifacts.append(path)
    def end_run(self): self.ended = True


def test_save_model_owns_run(monkeypatch, tmp_path):
    """save_model boundary: no run_id -> starts its own run, logs params + artifacts itself, ends run."""
    staging = tmp_path / "stage"; staging.mkdir()
    (staging / "model.pth").write_bytes(b"w")

    m = _OwnRunMlflow()
    client = _SaveClient()
    monkeypatch.setattr(reg, "_mlflow", lambda: m)
    monkeypatch.setattr(reg, "_client", lambda: client)

    rid = reg.save_model(staging, run_name="r", params={"model": {"ch": 8}})
    assert rid == "AUTOID"
    assert m.started and m.ended
    assert m.params == {"model.ch": 8}                  # flattened params logged
    assert m.artifacts == [str(staging / "model.pth")]  # own-run logs via mlflow, not client
