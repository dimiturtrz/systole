"""Geographic provenance tests (equivalence classes): country->continent SSOT + the M&Ms-1
centre-code map that turns the raw '1'-'5' codes into queryable country/site."""
from core.data.static.geo import COUNTRY_CONTINENT, continent_of
from core.data.static.mri.mnms1 import CENTRES


# --- continent_of: known / cross-continent / missing / unknown ---
def test_continent_known():
    assert continent_of("France") == "Europe"
    assert continent_of("China") == "Asia"
    assert continent_of("Canada") == "North America"


def test_continent_missing_or_unknown():
    assert continent_of(None) is None          # missing
    assert continent_of("Atlantis") is None    # not in the map


def test_every_dataset_country_has_a_continent():
    """Guard: the countries our adapters emit must all resolve (else continent silently nulls)."""
    for c in ("France", "Spain", "Germany", "Canada", "China"):
        assert continent_of(c) is not None and c in COUNTRY_CONTINENT


# --- M&Ms-1 centre map: code -> (site, country); the 'number -> place' matching ---
def test_mnms1_centre_codes_resolve_to_country():
    assert CENTRES["3"][1] == "Germany"            # Hamburg
    assert CENTRES["1"][1] == "Spain"              # Vall d'Hebron
    assert CENTRES["6"][1] == "Canada"             # McGill (test-only vendor)
    # every mapped centre resolves to a continent
    for _name, country in CENTRES.values():
        assert continent_of(country) is not None
