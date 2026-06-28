"""Geographic provenance helpers — the SSOT for country -> continent, so the adapters store a
canonical `country` and the continent (the Europe-vs-Asia analysis axis) is *derived*, never
hand-stored (one source, no drift). Add a dataset from a new country = one line here.
"""

# country -> continent. Only the countries our datasets actually come from (extend as needed).
COUNTRY_CONTINENT = {
    "France": "Europe",
    "Spain": "Europe",
    "Germany": "Europe",
    "Canada": "North America",
    "China": "Asia",
}


def continent_of(country: str | None) -> str | None:
    """Continent for a country name, or None if unknown/missing. Derive-on-read, don't store."""
    return COUNTRY_CONTINENT.get(country) if country else None
