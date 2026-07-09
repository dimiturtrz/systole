"""Geographic provenance tests (equivalence classes): country->continent SSOT + the M&Ms-1
centre-code map that turns the raw '1'-'5' codes into queryable country/site."""
from core.data.static.geo import COUNTRY_CONTINENT
from core.data.static.mri.mnms1 import CENTRES


# --- COUNTRY_CONTINENT map: known values + completeness for the countries our adapters emit ---
def test_continent_known():
    assert COUNTRY_CONTINENT["France"] == "Europe"
    assert COUNTRY_CONTINENT["China"] == "Asia"
    assert COUNTRY_CONTINENT["Canada"] == "North America"


def test_every_dataset_country_has_a_continent():
    """Guard: the countries our adapters emit must all resolve (else continent silently nulls)."""
    for c in ("France", "Spain", "Germany", "Canada", "China"):
        assert c in COUNTRY_CONTINENT


# --- M&Ms-1 centre map: code -> (site, country); the 'number -> place' matching ---
def test_mnms1_centre_codes_resolve_to_country():
    assert CENTRES["3"][1] == "Germany"            # Hamburg
    assert CENTRES["1"][1] == "Spain"              # Vall d'Hebron
    assert CENTRES["6"][1] == "Canada"             # McGill (test-only vendor)
    # every mapped centre resolves to a continent
    for _name, country in CENTRES.values():
        assert country in COUNTRY_CONTINENT
