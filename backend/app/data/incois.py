"""NetCDF-backed source: the INCOIS objectively-analysed Argo grid.

Reads the compact ``incois_grid.nc`` built by ``scripts/build_incois_grid.py``
(real objectively-analysed temperature & salinity from Argo floats, Indian
Ocean, monthly, 24 depth levels). It is a drop-in :class:`OceanDataSource`: same
:class:`FieldSlice` contract as the synthetic source, so routers and frontend
are unchanged.

The product only covers the Indian Ocean, but the viewer draws a full globe. So
each slice embeds the regional block into a *global* half-degree grid and leaves
every other cell ``None`` ("no data") — the globe stays whole with a luminous
real-data patch over the Indian Ocean. The product carries no currents, so this
source serves only temperature and salinity.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import xarray as xr

from .base import FieldSlice, _sample_indices

# Global half-degree grid the frontend renders the globe on. Ascending in both
# axes to match the mesh's expectations; the INCOIS block registers exactly onto
# it because both use X.5-degree cell centres.
_GLOBAL_LAT = np.round(np.arange(-89.5, 90.0, 1.0), 1)     # 180
_GLOBAL_LON = np.round(np.arange(-179.5, 180.0, 1.0), 1)   # 360


class IncoisGriddedSource:
    """Serves depth/time slices of the INCOIS Argo grid on a global mesh."""

    name = "INCOIS objectively-analysed Argo · Indian Ocean"

    _UNITS = {"temperature": "°C", "salinity": "PSU"}

    def __init__(self, path: str) -> None:
        self._ds = xr.open_dataset(path)
        self._depths = [float(d) for d in np.asarray(self._ds["depth"].values)]
        self._times = [self._iso(t) for t in np.asarray(self._ds["time"].values)]
        lat = np.asarray(self._ds["lat"].values, dtype=float)
        lon = np.asarray(self._ds["lon"].values, dtype=float)
        self._nlat, self._nlon = lat.size, lon.size
        # Where the regional block sits in the global grid (both ascending).
        self._row0 = int(np.rint(lat[0] - _GLOBAL_LAT[0]))     # -29.5-(-89.5)=60
        self._col0 = int(np.rint(lon[0] - _GLOBAL_LON[0]))     #  30.5-(-179.5)=210
        self._domains: dict[str, dict[str, float]] = {}

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _iso(t) -> str:
        """numpy datetime64 -> ISO-8601 UTC string, e.g. 2020-01-15T00:00:00Z."""
        dt = np.datetime64(t, "s").astype("datetime64[s]").astype(object)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # -- catalogue -----------------------------------------------------------
    def variables(self) -> list[str]:
        return ["temperature", "salinity"]

    def units(self, variable: str) -> str:
        try:
            return self._UNITS[variable]
        except KeyError:
            raise KeyError(variable)

    def depths(self) -> list[float]:
        return list(self._depths)

    def times(self) -> list[str]:
        return list(self._times)

    def domain(self, variable: str) -> dict[str, float]:
        """Absolute min/max across every depth and month (cached).

        A fixed scale keeps a value at the same colour as you scrub depth/time,
        so the deep, cold slices read blue and the warm surface reads hot without
        the legend jumping per slice.
        """
        if variable not in self._UNITS:
            raise KeyError(variable)
        if variable not in self._domains:
            arr = np.asarray(self._ds[variable].values)
            self._domains[variable] = {
                "min": round(float(np.nanmin(arr)), 3),
                "max": round(float(np.nanmax(arr)), 3),
            }
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
        if variable not in self._UNITS:
            raise KeyError(variable)
        di = self._nearest_depth_index(depth)
        ti = self._resolve_time_index(time)

        block = np.asarray(self._ds[variable].values[ti, di], dtype=float)
        full = np.full((_GLOBAL_LAT.size, _GLOBAL_LON.size), np.nan, dtype=float)
        full[self._row0:self._row0 + self._nlat,
             self._col0:self._col0 + self._nlon] = block

        values = [[FieldSlice.clean(v) for v in row] for row in full]
        finite = np.isfinite(block)
        vmin = round(float(block[finite].min()), 3) if finite.any() else None
        vmax = round(float(block[finite].max()), 3) if finite.any() else None

        return FieldSlice(
            variable=variable,
            units=self._UNITS[variable],
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

    def column(self, variable: str, time: str | None, max_levels: int = 12) -> dict:
        """A depth-stacked, bbox-trimmed volume for one timestep.

        Reads the native regional block straight from the dataset (no global
        null-padding) and crops it to the lat/lon box that holds any finite
        value, so the payload is the tight Indian-Ocean region rather than a
        mostly-empty global grid — cheap enough to re-fetch during animation.
        Returns the same shape as :func:`app.data.base.column_via_slices`.
        """
        if variable not in self._UNITS:
            raise KeyError(variable)
        ti = self._resolve_time_index(time)
        idxs = _sample_indices(len(self._depths), max_levels)

        # (k, nlat, nlon) regional cube — self._ds is already the IO block.
        cube = np.asarray(self._ds[variable].values[ti, idxs], dtype=float)
        finite_any = np.isfinite(cube).any(axis=0)          # widest (shallow) extent
        if finite_any.any():
            rows = np.where(finite_any.any(axis=1))[0]
            cols = np.where(finite_any.any(axis=0))[0]
            r0, r1 = int(rows[0]), int(rows[-1])
            c0, c1 = int(cols[0]), int(cols[-1])
        else:
            r0, r1, c0, c1 = 0, self._nlat - 1, 0, self._nlon - 1

        lat = np.asarray(self._ds["lat"].values, dtype=float)[r0:r1 + 1]
        lon = np.asarray(self._ds["lon"].values, dtype=float)[c0:c1 + 1]
        trimmed = cube[:, r0:r1 + 1, c0:c1 + 1]
        slices = [[[FieldSlice.clean(v) for v in row] for row in level]
                  for level in trimmed]
        finite = np.isfinite(trimmed)
        vmin = round(float(trimmed[finite].min()), 3) if finite.any() else None
        vmax = round(float(trimmed[finite].max()), 3) if finite.any() else None

        return {
            "variable": variable,
            "units": self._UNITS[variable],
            "time": self._times[ti],
            "time_index": ti,
            "depths": [float(self._depths[i]) for i in idxs],
            "lat": [float(v) for v in lat],
            "lon": [float(v) for v in lon],
            "shape": [len(lat), len(lon)],
            "value_min": vmin,
            "value_max": vmax,
            "slices": slices,
        }
