"""Unit tests for the shape-contract coverage gate (bd cardiac-seg-m8xq): a public method with a bare
array/tensor annotation is flagged; a jaxtyping-shaped one (or a dtype-only reduction) satisfies it."""
import subprocess
import sys
import textwrap
from pathlib import Path

from devtools.shape_contracts import analyze

_REPO = Path(__file__).resolve().parents[3]


def _write(tmp_path, src):
    f = tmp_path / "m.py"
    f.write_text(textwrap.dedent(src))
    return f


def test_bare_array_param_flagged(tmp_path):
    """A public method whose param is a bare `np.ndarray` / `Tensor` / array alias is surfaced."""
    rows = analyze(_write(tmp_path, """
        class M:
            @staticmethod
            def a(mask: np.ndarray) -> float: ...
            @staticmethod
            def b(x: Tensor, y: Mask): ...
    """))
    names = {name: slots for _, name, slots in rows}
    assert names["M.a"] == ["mask"]                 # bare ndarray param (the float return is not an array)
    assert names["M.b"] == ["x", "y"]               # Tensor + Mask alias both bare


def test_jaxtyping_annotation_satisfies(tmp_path):
    """A jaxtyping subscript (incl. dtype-only '...' and `| None`) is a satisfied contract — not flagged."""
    rows = analyze(_write(tmp_path, """
        class M:
            @staticmethod
            def a(mask: Integer[np.ndarray, "..."]) -> float: ...
            @staticmethod
            def b(x: Float[Tensor, "*b h w"], y: Float[Tensor, "b 1 h w"] | None = None): ...
    """))
    assert rows == []                               # every array slot carries a jaxtyping shape


def test_private_and_handlers_exempt(tmp_path):
    """Underscore-private helpers and CLI add_args/run handlers are interior/framework — not boundaries."""
    rows = analyze(_write(tmp_path, """
        class M:
            @staticmethod
            def _helper(mask: np.ndarray): ...
            @staticmethod
            def add_args(ap): ...
            @staticmethod
            def run(args: np.ndarray): ...
    """))
    assert rows == []


def test_nonarray_annotations_ignored(tmp_path):
    """A scalar/DataFrame/str param is not an array boundary — never flagged."""
    rows = analyze(_write(tmp_path, """
        class M:
            @staticmethod
            def a(n: int, name: str, df: pl.DataFrame) -> dict: ...
    """))
    assert rows == []


def test_assert_flag_exit_code(tmp_path):
    """The blocking gate: `--assert` exits 1 when a bare boundary remains, 0 once it's clean."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    def _run():
        return subprocess.run([sys.executable, "-m", "devtools.shape_contracts", str(pkg), "--assert"],  # noqa: S603
                              cwd=_REPO, capture_output=True).returncode
    (pkg / "m.py").write_text("class M:\n    @staticmethod\n    def a(mask: np.ndarray): ...\n")
    assert _run() == 1                                          # bare boundary -> gate fails
    (pkg / "m.py").write_text("class M:\n    @staticmethod\n    def a(n: int): ...\n")
    assert _run() == 0                                          # clean -> gate passes
