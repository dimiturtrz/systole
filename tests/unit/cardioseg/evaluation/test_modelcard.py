"""Per-run model-card rendering (modelcard) — the pure (config dict, metrics dict) -> markdown string
renderer, extracted from generate() (which reads config.json/metrics.json + probes the reference store —
the shell). Equivalence classes over the perf table (present/absent boundary distances) + card sections.
"""
from cardioseg.evaluation.modelcard import ModelCard


def _metrics(with_test=True):
    val = {"dice": {"LV-cav": 0.92, "LV-myo": 0.87, "RV": 0.90}, "dice_mean": 0.897,
           "boundary": {"LV-cav": {"hd95": 3.1, "assd": 0.8}}, "ef_mae": 4.2, "ef_rows": [1, 2, 3]}
    res = {"val": val}
    if with_test:
        res["test"] = {**val, "ef_mae": 6.1, "ef_rows": [1, 2]}
    return {"results": res, "n_train": 100, "n_val": 20}


def _cfg():
    return {"generator": {"data": {"sources": ["acdc"], "inplane": 1.5, "n4": False,
                                   "test_datasets": ["cmrxmotion"], "test_vendors": ["Canon", "GE"]}},
            "model": {"channels": [16, 32], "strides": [2], "res_units": 2, "out_channels": 4},
            "loss": {"kind": "dice_ce"}, "lr": 1e-3, "patience": 20, "epochs": 200, "seed": 0}


# --- _perf_table: boundary present vs missing ---
def test_perf_table_fills_boundary_when_present():
    """Present class: a class with boundary hd95/assd renders the numbers, not the em-dash."""
    r = {"dice": {"LV-cav": 0.9}, "dice_mean": 0.9, "boundary": {"LV-cav": {"hd95": 3.1, "assd": 0.8}}}
    t = ModelCard._perf_table(r)
    assert "0.900" in t and "3.10" in t and "0.80" in t


def test_perf_table_dashes_when_boundary_absent():
    """Missing class: no boundary dict -> em-dash placeholders, dice still shown."""
    r = {"dice": {"LV-cav": 0.9}, "dice_mean": 0.9}
    t = ModelCard._perf_table(r)
    assert "0.900" in t and "—" in t


def test_perf_table_skips_classes_not_in_dice():
    """A class absent from dice is omitted from the table body (no KeyError)."""
    r = {"dice": {"RV": 0.8}, "dice_mean": 0.8}
    t = ModelCard._perf_table(r)
    assert "RV" in t and "LV-cav" not in t.replace("bg/RV/myo/LV-cav", "")


# --- render_card: sections + val/test toggle ---
def test_render_card_has_core_sections_and_run_name():
    """Structure class: card names the run + has Model/Data/Performance/limitations sections."""
    md = ModelCard.render_card("myrun", _cfg(), _metrics())
    for h in ("Model Card — cardioseg run `myrun`", "## Model", "## Data & split",
              "## Performance", "## Intended use & limitations"):
        assert h in md


def test_render_card_includes_test_section_when_present():
    """Test-present class: a test axis -> a Held-out test heading appears."""
    assert "### Held-out test" in ModelCard.render_card("r", _cfg(), _metrics(with_test=True))


def test_render_card_omits_test_section_when_absent():
    """Test-absent class: val-only metrics -> no held-out-test heading."""
    assert "### Held-out test" not in ModelCard.render_card("r", _cfg(), _metrics(with_test=False))


def test_render_card_reads_flat_config_data():
    """Back-compat class: a flat config (no `generator`) still resolves data via the `cfg` fallback."""
    flat = _cfg()["generator"]["data"]
    cfg = {"data": flat, "model": _cfg()["model"], "loss": {"kind": "dice_ce"},
           "lr": 1e-3, "patience": 20, "epochs": 200, "seed": 0}
    md = ModelCard.render_card("r", cfg, _metrics())
    assert "acdc" in md


def test_render_card_appends_reference_section():
    """Ref class: a passed ref_section list is appended verbatim (the reference-ranges block)."""
    md = ModelCard.render_card(
        "r", _cfg(), _metrics(), ref_section=["", "## Reference ranges (derived from our GT, for context)"],
    )
    assert "## Reference ranges" in md


# --- reference_rows: provenance dict -> markdown bullets (unit/skip classes) ---
def test_reference_rows_renders_value_unit_and_provenance():
    """Present class: EF renders '%' + n + based_on; volume keys render ' mL'."""
    prov = {"ef_normal": {"value": [55, 70], "n": 120, "based_on": "cohortX"},
            "edv_normal_ml": {"value": [90, 150], "n": 80, "based_on": "cohortY"}}
    rows = ModelCard.reference_rows(prov)
    assert any("55–70%" in r and "n=120" in r and "cohortX" in r for r in rows)
    assert any("90–150 mL" in r for r in rows)


def test_reference_rows_skips_missing_or_scalar_value():
    """Skip class: a missing key or a non-list value is dropped (only [lo,hi] lists render)."""
    prov = {"ef_normal": None, "edv_normal_ml": {"value": 100}, "esv_normal_ml": {"value": [30, 60]}}
    rows = ModelCard.reference_rows(prov)
    assert len(rows) == 1 and "30–60 mL" in rows[0]


def test_reference_rows_empty_when_no_provenance():
    """Empty class: no provenances -> no rows (the section is then omitted upstream)."""
    assert ModelCard.reference_rows({}) == []
