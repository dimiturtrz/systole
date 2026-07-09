"""Visual synth-vs-real diagnostic (core.data.analysis.render). A plotting CLI (coverage-omitted): its
one method (render_synth_vs_real) loads real val slices, synthesizes, and writes a PNG grid — pure I/O
+ rendering, no isolable helper. THIN mirror — assert the class + method exist so the module imports
cleanly and its surface stays intact (the visual check itself is a run-time smoke test)."""
from core.data.analysis.render import Render


def test_render_surface():
    """The renderer staticmethod exists on the class (structural mirror; the figure it writes is a
    run-time diagnostic, not a unit assertion)."""
    assert callable(Render.render_synth_vs_real)
