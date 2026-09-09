"""Synthetic in-situ instruments (Argo floats, Gliders, CTD casts).

Phase-3 counterpart to the gridded model fields: a small, deterministic set of
instruments scattered across the Indian Ocean (INCOIS area of interest), each
carrying a depth profile of temperature / salinity for the profile-chart panel.

Profiles are sampled from the *same* analytic fields as :mod:`.synthetic`, so an
instrument sitting at a grid point agrees with the model surface it's drawn on —
they tell one consistent story. Swap this for real Argo GDAC / glider files
behind the same method signatures and nothing downstream changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .synthetic import DEPTH_LEVELS, SyntheticOceanSource

# Instrument platform kinds. Each renders as a distinct marker on the globe.
KINDS = ("argo", "glider", "ctd")

# Which variables each platform reports (drives the profile chart's series).
_KIND_VARIABLES = {
    "argo": ("temperature", "salinity"),
    "glider": ("temperature", "salinity"),
    "ctd": ("temperature", "salinity"),
}

# How deep each platform profiles (metres). Argo floats reach ~2000 m; gliders
# work the upper ocean; CTD casts here are shelf/upper-ocean.
_KIND_MAX_DEPTH = {"argo": 2000.0, "glider": 1000.0, "ctd": 500.0}


@dataclass(frozen=True)
class Instrument:
    """One deployed platform at a fixed position (latest known location)."""

    id: str
    kind: str
    name: str
    lat: float
    lon: float
    variables: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "variables": list(self.variables),
        }


# Fixed roster: (id, kind, name, lat, lon). Positions sit in the Indian Ocean
# so they overlay the INCOIS region a user is most likely to inspect.
_ROSTER: list[tuple[str, str, str, float, float]] = [
    ("argo-2901623", "argo", "Argo 2901623", -5.0, 65.0),
    ("argo-2902114", "argo", "Argo 2902114", 8.0, 72.0),
    ("argo-2902677", "argo", "Argo 2902677", -14.0, 88.0),
    ("argo-5906312", "argo", "Argo 5906312", 2.0, 54.0),
    ("glider-sg562", "glider", "Glider SG562", 12.0, 68.0),
    ("glider-sg118", "glider", "Glider SG118", -2.0, 80.0),
    ("ctd-inx044", "ctd", "CTD INX-044", 15.0, 74.0),
    ("ctd-inx071", "ctd", "CTD INX-071", -20.0, 60.0),
]


class InstrumentSource:
    """Roster of instruments + depth profiles sampled from the model fields."""

    name = "synthetic"

    def __init__(self, ocean: SyntheticOceanSource | None = None) -> None:
        # Reuse the gridded source so profiles match the rendered field exactly.
        self._ocean = ocean or SyntheticOceanSource()
        self._instruments = [
            Instrument(
                id=i, kind=k, name=n, lat=la, lon=lo,
                variables=_KIND_VARIABLES[k],
            )
            for (i, k, n, la, lo) in _ROSTER
        ]
        self._by_id = {inst.id: inst for inst in self._instruments}

    # -- catalogue -----------------------------------------------------------
    def list(self, bbox: tuple[float, float, float, float] | None = None) -> list[Instrument]:
        """All instruments, optionally filtered to ``(min_lon,min_lat,max_lon,max_lat)``."""
        if bbox is None:
            return list(self._instruments)
        min_lon, min_lat, max_lon, max_lat = bbox
        return [
            inst for inst in self._instruments
            if min_lon <= inst.lon <= max_lon and min_lat <= inst.lat <= max_lat
        ]

    def get(self, instrument_id: str) -> Instrument:
        return self._by_id[instrument_id]

    # -- profiles ------------------------------------------------------------
    def profile(self, instrument_id: str, variable: str, time: str | None = None) -> dict:
        """Depth profile of ``variable`` at the instrument's location.

        Samples the analytic field at each standard depth (down to the platform's
        max depth) using the grid cell nearest the instrument. Returns paired
        ``depths`` / ``values`` arrays ready for a depth-vs-value line chart.
        """
        inst = self._by_id.get(instrument_id)
        if inst is None:
            raise KeyError(instrument_id)
        if variable not in inst.variables:
            raise KeyError(variable)

        max_depth = _KIND_MAX_DEPTH.get(inst.kind, 2000.0)
        depths = [float(d) for d in DEPTH_LEVELS if d <= max_depth]

        row, col = self._nearest_cell(inst.lat, inst.lon)
        units = self._ocean.units(variable)
        t_index = self._ocean._resolve_time(time)

        values: list[float | None] = []
        for d in depths:
            grid = self._ocean._fields[variable][1](d, t_index)
            values.append(round(float(grid[row, col]), 3))

        return {
            "id": inst.id,
            "kind": inst.kind,
            "name": inst.name,
            "lat": inst.lat,
            "lon": inst.lon,
            "variable": variable,
            "units": units,
            "time": self._ocean.times()[t_index],
            "depths": depths,
            "values": values,
        }

    # -- helpers -------------------------------------------------------------
    def _nearest_cell(self, lat: float, lon: float) -> tuple[int, int]:
        """Index of the grid cell nearest ``(lat, lon)``."""
        import numpy as np

        row = int(np.argmin(np.abs(self._ocean.lat - lat)))
        col = int(np.argmin(np.abs(self._ocean.lon - lon)))
        return row, col
