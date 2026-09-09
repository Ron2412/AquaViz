"""NetCDF-backed source: the Copernicus GLORYS global physics reanalysis.

Reads the compact ``glorys_grid.nc`` built by ``scripts/build_glorys_grid.py``
from Copernicus **GLOBAL_MULTIYEAR_PHY_001_030** (GLORYS12V1): global potential
temperature (``thetao``), salinity (``so``) and the horizontal current vector
(``uo`` / ``vo``), regridded onto the viewer's global 1° mesh. It is a drop-in
:class:`~app.data.base.OceanDataSource` — same :class:`FieldSlice` contract as
the synthetic and INCOIS sources — so routers and frontend are unchanged.

Two things set it apart from :mod:`app.data.incois`:

* **Global, no embedding.** GLORYS already covers the whole ocean, so the built
  grid *is* the global 1° grid; a slice is read straight off it (INCOIS instead
  embeds a regional Indian-Ocean block into an otherwise-empty globe).
* **Currents.** Beyond ``temperature`` and ``salinity`` it serves
  ``current_speed`` — the magnitude ``hypot(uo, vo)`` of the current vector —
  as a scalar field that renders with the existing neon colormap. The ``uo`` /
  ``vo`` components are kept in the file so a future arrow/streamline layer can
  read them without re-downloading.

Land / missing cells are ``NaN`` in the file and become ``None`` in the slice,
so the holographic globe shows dark land with a luminous data layer over ocean.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import xarray as xr

from .base import FieldSlice

# Canonical global 1° mesh the frontend renders on — identical to the grid the
# INCOIS source targets, so GLORYS is a drop-in on the very same globe geometry.
_GLOBAL_LAT = np.round(np.arange(-89.5, 90.0, 1.0), 1)     # 180
_GLOBAL_LON = np.round(np.arange(-179.5, 180.0, 1.0), 1)   # 360

# Scalar variables served to the viewer. current_speed is derived from uo/vo;
# the others are read directly. Units drive the legend + profile axis labels.
_UNITS = {"temperature": "°C", "salinity": "PSU", "current_speed": "m/s"}


class GlorysGriddedSource:
    """Serves depth/time slices of the GLORYS global grid, currents included."""

    name = "Copernicus GLORYS12V1 reanalysis · Global"

    def __init__(self, path: str) -> None:
        self._ds = xr.open_dataset(path)
        self._depths = [float(d) for d in np.asarray(self._ds["depth"].values)]
        self._times = [self._iso(t) for t in np.asarray(self._ds["time"].values)]
        # Which stored variables are present decides whether currents are served.
        self._has_currents = "uo" in self._ds and "vo" in self._ds
        self._domains: dict[str, dict[str, float]] = {}

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _iso(t) -> str:
        """numpy datetime64 -> ISO-8601 UTC string, e.g. 2021-01-15T00:00:00Z."""
        dt = np.datetime64(t, "s").astype("datetime64[s]").astype(object)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def _speed(self, ti: int, di: int) -> np.ndarray:
        """Current-speed plane = |(uo, vo)| at one time/depth (NaN where either is)."""
        uo = np.asarray(self._ds["uo"].values[ti, di], dtype=float)
        vo = np.asarray(self._ds["vo"].values[ti, di], dtype=float)
        return np.hypot(uo, vo)

    # -- catalogue -----------------------------------------------------------
    def variables(self) -> list[str]:
        out = ["temperature", "salinity"]
        if self._has_currents:
            out.append("current_speed")
        return out

    def units(self, variable: str) -> str:
        if variable == "current_speed" and not self._has_currents:
            raise KeyError(variable)
        try:
            return _UNITS[variable]
        except KeyError:
            raise KeyError(variable)

    def depths(self) -> list[float]:
        return list(self._depths)

    def times(self) -> list[str]:
        return list(self._times)

    def domain(self, variable: str) -> dict[str, float]:
        """Absolute min/max across every depth and time (cached).

        A fixed scale keeps a value the same colour as you scrub depth/time, so
        deep cold water reads blue and the warm surface reads hot without the
        legend jumping per slice. The compact file is already downsampled, so
        loading a variable whole here is cheap.
        """
        if variable not in self.variables():
            raise KeyError(variable)
        if variable not in self._domains:
            if variable == "current_speed":
                # Per-time to bound memory; magnitude is non-negative.
                lo, hi = np.inf, -np.inf
                for ti in range(len(self._times)):
                    for di in range(len(self._depths)):
                        arr = self._speed(ti, di)
                        if np.isfinite(arr).any():
                            lo = min(lo, float(np.nanmin(arr)))
                            hi = max(hi, float(np.nanmax(arr)))
                if not np.isfinite(lo):
                    lo, hi = 0.0, 0.0
            else:
                arr = np.asarray(self._ds[variable].values)
                lo = float(np.nanmin(arr))
                hi = float(np.nanmax(arr))
            self._domains[variable] = {"min": round(lo, 3), "max": round(hi, 3)}
        return dict(self._domains[variable])

    def _nearest_depth_index(self, depth: float) -> int:
        arr = np.asarray(self._depths)
        return int(np.argmin(np.abs(arr - float(depth))))

    def _resolve_time_index(self, time: str | None) -> int:
        if not time:
            return 0
        # A '+' in a URL-decoded ISO offset can arrive as a space.
        candidates = [time]
        if " " in time:
            candidates.append(time.replace(" ", "+"))
        for cand in candidates:
            if cand in self._times:
                return self._times.index(cand)

        def _parse(s: str) -> datetime | None:
            try:
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
            except ValueError:
                return None

        target = next((_parse(c) for c in candidates if _parse(c)), None)
        if target is None:
            return 0
        parsed = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in self._times]
        return min(range(len(parsed)), key=lambda i: abs(parsed[i] - target))

    # -- OceanDataSource -----------------------------------------------------
    def slice(self, variable: str, depth: float, time: str | None) -> FieldSlice:
        if variable not in self.variables():
            raise KeyError(variable)
        di = self._nearest_depth_index(depth)
        ti = self._resolve_time_index(time)

        if variable == "current_speed":
            block = self._speed(ti, di)
        else:
            block = np.asarray(self._ds[variable].values[ti, di], dtype=float)

        # GLORYS is already global on the 1° grid; no regional embedding.
        values = [[FieldSlice.clean(v) for v in row] for row in block]
        finite = np.isfinite(block)
        vmin = round(float(block[finite].min()), 3) if finite.any() else None
        vmax = round(float(block[finite].max()), 3) if finite.any() else None

        return FieldSlice(
            variable=variable,
            units=_UNITS[variable],
            depth=float(self._depths[di]),
            depth_requested=float(depth),
            time=self._times[ti],
            time_index=ti,
            lat=[float(v) for v in _GLOBAL_LAT],
            lon=[float(v) for v in _GLOBAL_LON],
            values=values,
            value_min=vmin,
            value_max=vmax,
        )
