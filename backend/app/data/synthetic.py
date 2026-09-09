"""Synthetic ocean fields (pure NumPy) for building the pipeline offline.

Interim :class:`OceanDataSource` used while network access for downloading real
NetCDF is unavailable. It produces physically plausible *global* fields on a
regular lat/lon/depth grid for several variables:

* **temperature** — warm equatorial surface (~29 degC) cooling toward the poles,
  an exponential thermocline decaying to a cold (~4 degC) abyss, plus a warm
  mesoscale eddy for visual interest.
* **salinity** — salty subtropical gyres (~37 PSU), fresher equator and poles,
  rising slightly through the halocline.
* **current_speed** — fast equatorial and western-boundary flow (~1.5 m/s)
  decaying with depth toward a near-still deep ocean.

All vary gently timestep-to-timestep. Replace with a NetCDF-backed source for
real data; the router contract (:class:`FieldSlice`) is identical, so nothing
downstream changes.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Callable

import numpy as np

from .base import FieldSlice

# --- Grid definition: whole globe (for the 3D globe viewer) ---------------
# Longitude excludes +180 so the mesh can wrap the antimeridian seam without a
# duplicated column; latitude includes both poles.
LON_MIN, LON_MAX, LON_STEP = -180.0, 180.0, 1.0   # 180W .. 179E
LAT_MIN, LAT_MAX, LAT_STEP = -90.0, 90.0, 1.0     # 90S .. 90N

# Standard oceanographic depth levels (metres).
DEPTH_LEVELS: list[float] = [
    0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000
]

# Fixed start date keeps output deterministic (no wall-clock dependency).
_START = datetime(2026, 9, 1, tzinfo=timezone.utc)
_N_TIMES = 5  # daily steps


class SyntheticOceanSource:
    """Generates analytic slices for several variables on a lat/lon/depth grid."""

    name = "synthetic"

    def __init__(self) -> None:
        self.lat = np.round(
            np.arange(LAT_MIN, LAT_MAX + LAT_STEP / 2, LAT_STEP), 3
        )
        # Stop *before* LON_MAX so +180 is excluded (it wraps back to -180).
        self.lon = np.round(np.arange(LON_MIN, LON_MAX, LON_STEP), 3)
        # Precompute 2-D coordinate grids (rows=lat, cols=lon).
        self._lat2d, self._lon2d = np.meshgrid(self.lat, self.lon, indexing="ij")
        self._times = [
            (_START + timedelta(days=i)).isoformat() for i in range(_N_TIMES)
        ]

        # Variable registry: name -> (units, field function). Adding a variable
        # is one entry here; routers and the frontend discover it via /meta/coords.
        self._fields: dict[str, tuple[str, Callable[[float, int], np.ndarray]]] = {
            "temperature": ("degree_C", self._temperature),
            "salinity": ("PSU", self._salinity),
            "current_speed": ("m/s", self._current_speed),
        }
        self._domains: dict[str, dict[str, float]] = {}  # lazily cached

    # -- catalogue -----------------------------------------------------------
    def variables(self) -> list[str]:
        return list(self._fields)

    def units(self, variable: str) -> str:
        try:
            return self._fields[variable][0]
        except KeyError:
            raise KeyError(variable)

    def depths(self) -> list[float]:
        return [float(d) for d in DEPTH_LEVELS]

    def times(self) -> list[str]:
        return list(self._times)

    def domain(self, variable: str) -> dict[str, float]:
        """Fixed value range for ``variable`` across every depth and timestep.

        The frontend colours every slice against this absolute scale so a given
        value maps to the same colour at every depth (otherwise each slice
        self-normalises and the colours/legend jump as you scrub depth/time).
        Computed once per variable, then cached.
        """
        if variable not in self._fields:
            raise KeyError(variable)
        if variable not in self._domains:
            field = self._fields[variable][1]
            lo, hi = float("inf"), float("-inf")
            for t_index in range(len(self._times)):
                for depth in DEPTH_LEVELS:
                    grid = field(float(depth), t_index)
                    lo = min(lo, float(np.nanmin(grid)))
                    hi = max(hi, float(np.nanmax(grid)))
            self._domains[variable] = {"min": round(lo, 3), "max": round(hi, 3)}
        return dict(self._domains[variable])

    # -- field models --------------------------------------------------------
    def _temperature(self, depth: float, t_index: int) -> np.ndarray:
        """Analytic temperature (degC) over the whole grid at one depth/time."""
        lat2d, lon2d = self._lat2d, self._lon2d

        # Sea-surface temperature: Gaussian in latitude, warm at the equator.
        surface = 5.0 + 24.0 * np.exp(-(lat2d ** 2) / (2 * 15.0 ** 2))

        # Exponential thermocline toward a cold abyss.
        thermocline = math.exp(-depth / 500.0)
        temp = 4.0 + (surface - 4.0) * thermocline

        # Zonal ripple + a warm mesoscale eddy, both fading with depth.
        temp += 1.5 * np.sin(np.radians(lon2d * 3.0)) * math.exp(-depth / 200.0)
        eddy = np.exp(-(((lat2d - 10.0) ** 2 + (lon2d - 70.0) ** 2) / (2 * 6.0 ** 2)))
        temp += 2.0 * eddy * math.exp(-depth / 150.0)

        # Gentle temporal drift.
        temp += 0.5 * math.sin(t_index * 0.7)
        return temp

    def _salinity(self, depth: float, t_index: int) -> np.ndarray:
        """Analytic salinity (PSU) — salty subtropics, fresh equator/poles."""
        lat2d, lon2d = self._lat2d, self._lon2d

        # Twin subtropical salinity maxima (~+/-25 deg), fresher at the equator
        # (rain belt) and toward the poles.
        subtropics = np.exp(-((np.abs(lat2d) - 25.0) ** 2) / (2 * 12.0 ** 2))
        surface = 34.0 + 3.0 * subtropics - 0.8 * np.exp(-(lat2d ** 2) / (2 * 6.0 ** 2))

        # Halocline: salinity rises a little with depth toward a stable deep value.
        deep = 34.9
        sal = deep + (surface - deep) * math.exp(-depth / 400.0)

        # Faint zonal structure + tiny temporal drift.
        sal += 0.15 * np.sin(np.radians(lon2d * 2.0)) * math.exp(-depth / 300.0)
        sal += 0.05 * math.sin(t_index * 0.7 + 1.0)
        return sal

    def _current_speed(self, depth: float, t_index: int) -> np.ndarray:
        """Analytic current speed (m/s) — fast equator/west boundaries, slow deep."""
        lat2d, lon2d = self._lat2d, self._lon2d

        # Equatorial jet: strong near the equator, decaying poleward.
        equatorial = 1.2 * np.exp(-(lat2d ** 2) / (2 * 6.0 ** 2))

        # Western-boundary intensification: faster on the western side of basins.
        western = 0.5 * (0.5 + 0.5 * np.cos(np.radians(lon2d))) \
            * np.exp(-((np.abs(lat2d) - 35.0) ** 2) / (2 * 12.0 ** 2))

        # Mesoscale swirl for texture.
        swirl = 0.25 * np.abs(np.sin(np.radians(lat2d * 4.0 + lon2d * 2.0)))

        surface = 0.05 + equatorial + western + swirl
        # Currents weaken with depth toward a near-still abyss.
        speed = surface * math.exp(-depth / 250.0) + 0.02
        # Gentle temporal pulse.
        speed *= 1.0 + 0.05 * math.sin(t_index * 0.7 + 0.5)
        return speed

    # -- snapping helpers ----------------------------------------------------
    def _nearest_depth(self, depth: float) -> float:
        levels = self.depths()
        return min(levels, key=lambda d: abs(d - depth))

    def _resolve_time(self, time: str | None) -> int:
        if time is None:
            return 0
        # A '+' in an ISO-8601 UTC offset can arrive URL-decoded as a space
        # (e.g. "...00:00 00:00"); try the raw value and a '+'-restored variant.
        candidates = [time]
        if " " in time:
            candidates.append(time.replace(" ", "+"))
        for cand in candidates:
            if cand in self._times:
                return self._times.index(cand)
        target: datetime | None = None
        for cand in candidates:
            try:
                target = datetime.fromisoformat(cand)
                break
            except ValueError:
                continue
        if target is None:
            return 0
        parsed = [datetime.fromisoformat(t) for t in self._times]
        return min(range(len(parsed)), key=lambda i: abs(parsed[i] - target))

    # -- OceanDataSource -----------------------------------------------------
    def slice(self, variable: str, depth: float, time: str | None) -> FieldSlice:
        try:
            units, field = self._fields[variable]
        except KeyError:
            raise KeyError(variable)

        actual_depth = self._nearest_depth(depth)
        t_index = self._resolve_time(time)
        grid = field(actual_depth, t_index)

        values = [[FieldSlice.clean(v) for v in row] for row in grid]
        return FieldSlice(
            variable=variable,
            units=units,
            depth=actual_depth,
            depth_requested=float(depth),
            time=self._times[t_index],
            time_index=t_index,
            lat=[float(v) for v in self.lat],
            lon=[float(v) for v in self.lon],
            values=values,
            value_min=round(float(np.nanmin(grid)), 3),
            value_max=round(float(np.nanmax(grid)), 3),
        )
