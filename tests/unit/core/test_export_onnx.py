"""core.export_onnx tests (thin): the real ONNX export is coverage-omitted (heavy onnxruntime +
quantization, gated on a real subject) and smoke-tested end to end by the entry-point tests.

Here we only assert the module imports and its public surface exists/is callable — a structural
mirror so a rename or a dropped method breaks a test. onnxruntime may be absent on some lanes
(Windows CPU fallback), so the import is skip-guarded.
"""
import pytest

pytest.importorskip("onnxruntime")

from core.export_onnx import ExportOnnx


def test_export_surface_is_callable():
    """The parity-gated exporter + its helpers + CLI surface are present and callable."""
    assert callable(ExportOnnx.export)
    assert callable(ExportOnnx.run)
    assert callable(ExportOnnx.add_args)
    assert callable(ExportOnnx._parity)
    assert callable(ExportOnnx._load_model)
