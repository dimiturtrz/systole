"""Mirror test for the export dispatcher (bd cardiac-seg-6e1a)."""
from core import export


def test_commands_registered():
    assert set(export.COMMANDS) == {"onnx", "mesh"}


def test_each_command_has_run_and_add_args():
    for cls in export.COMMANDS.values():
        assert hasattr(cls, "run") and hasattr(cls, "add_args")
