"""Country centroids for the results map (approximate, decimal degrees)."""
from __future__ import annotations

COUNTRY_CENTROIDS: dict[str, tuple[float, float]] = {
    "AT": (47.5, 14.5),
    "BE": (50.8, 4.5),
    "CZ": (49.8, 15.5),
    "DE": (51.2, 10.4),
    "FR": (46.6, 2.2),
    "HR": (45.1, 15.2),
    "HU": (47.2, 19.5),
    "LU": (49.8, 6.1),
    "NL": (52.1, 5.3),
    "PL": (51.9, 19.1),
    "RO": (45.9, 24.9),
    "SI": (46.1, 14.8),
    "SK": (48.7, 19.7),
}

DEFAULT_CENTER = (
    round(sum(lat for lat, _ in COUNTRY_CENTROIDS.values()) / len(COUNTRY_CENTROIDS), 2),
    round(sum(lon for _, lon in COUNTRY_CENTROIDS.values()) / len(COUNTRY_CENTROIDS), 2),
)
DEFAULT_ZOOM = 5
