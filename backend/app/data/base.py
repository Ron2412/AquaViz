"""Abstract interface between the API and its ocean data.

Routers talk to data only through :class:`OceanDataSource`, so the backing store
can be swapped without touching endpoint code: synthetic NumPy fields now (while
network access for downloading real NetCDF is unavailable), and a NetCDF-backed
source (xarray + netCDF4) once real INCOIS / public files are on disk. Both must
return the same :class:`FieldSlice`, which is exactly what the frontend mesh
needs — a 2-D lat×lon grid of one variable at one depth and time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class FieldSlice:
    """A single variable sliced to one depth level and one timestep.

    ``values`` is a ``nlat x nlon`` grid aligned to ``lat`` (rows) and ``lon``
    (columns). Missing / land cells are ``None`` so the payload stays valid JSON
    (which cannot represent NaN) and the frontend can skip them.
    """

    variable: str
    units: str
    depth: float            # actual depth level served (metres)
    depth_requested: float  # depth the client asked for (before snapping)
    time: str               # actual timestep served (ISO-8601)
    time_index: int
    lat: list[float]
    lon: list[float]
    values: list[list[float | None]]
    value_min: float | None
    value_max: float | None

    def to_dict(self) -> dict:
        return {
            "variable": self.variable,
            "units": self.units,
            "depth": self.depth,
            "depth_requested": self.depth_requested,
            "time": self.time,
            "time_index": self.time_index,
            "lat": self.lat,
            "lon": self.lon,
            "shape": [len(self.lat), len(self.lon)],
            "values": self.values,
            "value_min": self.value_min,
            "value_max": self.value_max,
        }

    @staticmethod
    def clean(value: float | None) -> float | None:
        """Coerce a cell to a JSON-safe rounded float, or None if missing."""
        if value is None:
            return None
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return round(value, 3)


@runtime_checkable
class OceanDataSource(Protocol):
    """Everything a router needs to serve sliced fields and build UI controls."""

    name: str

    def variables(self) -> list[str]:
        """Names of variables this source can serve (e.g. ``"temperature"``)."""
        ...

    def depths(self) -> list[float]:
        """Available depth levels in metres, ascending."""
        ...

    def times(self) -> list[str]:
        """Available timesteps as ISO-8601 strings, ascending."""
        ...

    def units(self, variable: str) -> str:
        """Units string for ``variable`` (e.g. ``"degree_C"``)."""
        ...

    def domain(self, variable: str) -> dict[str, float]:
        """Fixed ``{"min", "max"}`` range for ``variable`` across all depths/times.

        Lets the frontend colour every slice on one absolute scale instead of
        self-normalising per slice.
        """
        ...

    def slice(
        self, variable: str, depth: float, time: str | None
    ) -> FieldSlice:
        """Return one depth/time slice of ``variable`` as a 2-D lat×lon grid.

        Implementations snap ``depth`` and ``time`` to the nearest available
        level/step and report the actual values on the returned slice.
        """
        ...

    # Optional extension (feature-detected by the router, not required):
    #
    #   def column(self, variable, time, max_levels) -> dict
    #
    # returns a depth-stacked, bbox-trimmed volume for one timestep (the shape
    # produced by :func:`column_via_slices`). Sources backed by a compact
    # regional store should implement it directly for speed; everyone else is
    # served by the generic fallback below.


def _sample_indices(n: int, k: int) -> list[int]:
    """Up to ``k`` indices spread evenly across ``range(n)``, incl. 0 and n-1."""
    if n <= 0:
        return []
    if n <= k:
        return list(range(n))
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def column_via_slices(
    source, variable: str, time: str | None, max_levels: int = 12
) -> dict:
    """Depth-stacked, bbox-trimmed volume built from repeated ``slice()`` calls.

    Generic fallback that works with any :class:`OceanDataSource`: it samples up
    to ``max_levels`` depth levels (always including the shallowest and deepest),
    slices each, crops every level to the lat/lon bounding box holding any finite
    data, and returns a JSON-ready stack shaped like a per-source ``column()``
    override. Sources backed by a compact regional store (e.g. INCOIS) should
    implement ``column()`` directly for speed; this keeps results identical for
    the rest. A spatial stride caps very large (e.g. global, dense) grids so the
    payload stays bounded.
    """
    depths = source.depths()
    idxs = _sample_indices(len(depths), max_levels)
    slices = [source.slice(variable, depth=depths[i], time=time) for i in idxs]
    first = slices[0]
    lat, lon = first.lat, first.lon
    nlat, nlon = len(lat), len(lon)

    # Union bounding box of finite cells across every sampled level.
    r0, r1, c0, c1 = nlat, -1, nlon, -1
    for sl in slices:
        for j, row in enumerate(sl.values):
            for i, v in enumerate(row):
                if v is not None:
                    if j < r0:
                        r0 = j
                    if j > r1:
                        r1 = j
                    if i < c0:
                        c0 = i
                    if i > c1:
                        c1 = i
    if r1 < r0:                        # no finite data anywhere
        r0, r1, c0, c1 = 0, nlat - 1, 0, nlon - 1

    # Cap payload for dense global grids with a spatial stride.
    budget = 60_000                    # cells per level (≈ INCOIS regional size)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    stride = 1
    while (-(-h // stride)) * (-(-w // stride)) > budget:
        stride += 1

    rows = list(range(r0, r1 + 1, stride))
    cols = list(range(c0, c1 + 1, stride))
    out_slices = [[[sl.values[j][i] for i in cols] for j in rows] for sl in slices]
    vmins = [sl.value_min for sl in slices if sl.value_min is not None]
    vmaxs = [sl.value_max for sl in slices if sl.value_max is not None]
    return {
        "variable": variable,
        "units": first.units,
        "time": first.time,
        "time_index": first.time_index,
        "depths": [float(depths[i]) for i in idxs],
        "lat": [float(lat[j]) for j in rows],
        "lon": [float(lon[i]) for i in cols],
        "shape": [len(rows), len(cols)],
        "value_min": min(vmins) if vmins else None,
        "value_max": max(vmaxs) if vmaxs else None,
        "slices": out_slices,
    }
