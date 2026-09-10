"""HYCOM ocean-model reader — the *model* side of the comparison platform.

Wraps a large (~10 GB) HYCOM NetCDF as an :class:`~app.data.base.OceanDataSource`
without ever loading the whole file. The dataset is opened once (headers + the
small coordinate arrays only); each :meth:`slice` reads a single strided
depth/time plane straight from disk via xarray's lazy indexing. Raw NetCDF never
leaves the server — the browser only receives the coarsened 2-D grid it renders.

The structure was *inspected* (``scripts/inspect_hycom.py``), never assumed::

    dims   TIME=28 (6-hourly forecast), DEPTH=6 (0..500 m), LAT=1384, LON=1665
           (~0.06deg, Indian-Ocean domain 20..120degE, 45degS..31degN)
    vars   TEMP, SALN  (Argo-comparable), UVEL/VVEL (currents), SSH, MLD, TCHP,
           TEMP_CT (conservative T), SALNA (salinity anomaly)
    coords TIME datetime64[ns], DEPTH/LAT/LON float32

Only the Argo-comparable in-situ scalars are surfaced — ``TEMP`` -> temperature,
``SALN`` -> salinity — matching the existing UI and the temperature/salinity
comparison (not the conservative-temperature / anomaly fields). The file carries
no ``units`` attribute on TEMP/SALN; the HYCOM convention is degree_C / PSU.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .base import FieldSlice

# canonical variable name -> (HYCOM variable, units). Units follow the HYCOM
# convention; the file omits them on TEMP/SALN. Currents (UVEL/VVEL) are present
# and readable, but not surfaced here — the scalar UI and the Argo comparison
# only use temperature/salinity. Adding "current_speed" later is one entry here.
_VARIABLES: dict[str, tuple[str, str]] = {
    "temperature": ("TEMP", "degree_C"),
    "salinity": ("SALN", "PSU"),
}

# Coarsen the dense (~0.06deg) planes so the JSON the browser receives stays
# small: the larger axis is strided down to about this many points. The
# comparison then matches each float to the nearest coarsened cell — the
# ``distance_km`` it returns makes that matchup scale explicit and honest.
_MAX_AXIS = 220


class HycomModelSource:
    """Read-only HYCOM source: one open dataset handle, lazy per-slice reads."""

    def __init__(self, path: str) -> None:
        # Lazy open: netCDF4 backend reads headers + coordinate arrays only.
        self._ds = xr.open_dataset(path, decode_times=True)
        self.path = path

        # Coordinate arrays are 1-D and small — safe to hold in memory.
        self._depth = [round(float(d), 1) for d in np.asarray(self._ds["DEPTH"].values)]
        self._time_dt = np.asarray(self._ds["TIME"].values)  # datetime64[ns]
        self._time_iso = [np.datetime_as_string(t, unit="s") for t in self._time_dt]

        nlat = int(self._ds.sizes["LAT"])
        nlon = int(self._ds.sizes["LON"])
        self._stride = max(1, int(np.ceil(max(nlat, nlon) / _MAX_AXIS)))
        self._lat = [round(float(v), 4)
                     for v in np.asarray(self._ds["LAT"].values)[:: self._stride]]
        self._lon = [round(float(v), 4)
                     for v in np.asarray(self._ds["LON"].values)[:: self._stride]]

        start = self._time_iso[0][:10] if self._time_iso else "?"
        self.name = f"HYCOM RSMC · Indian Ocean forecast ({start})"
        self._domain_cache: dict[str, dict] = {}

    # -- catalogue -------------------------------------------------------
    def variables(self) -> list[str]:
        return list(_VARIABLES)

    def depths(self) -> list[float]:
        return list(self._depth)

    def times(self) -> list[str]:
        return list(self._time_iso)

    def units(self, variable: str) -> str:
        try:
            return _VARIABLES[variable][1]
        except KeyError:
            raise KeyError(variable)

    def _hvar(self, variable: str) -> str:
        try:
            return _VARIABLES[variable][0]
        except KeyError:
            raise KeyError(variable)  # router maps to 404 / comparison re-raises

    # -- index snapping --------------------------------------------------
    def _depth_index(self, depth: float) -> int:
        arr = np.asarray(self._depth, dtype=float)
        return int(np.argmin(np.abs(arr - float(depth))))

    def _time_index(self, time: str | None) -> int:
        if not time or self._time_dt.size == 0:
            return 0
        try:
            target = np.datetime64(str(time).replace("Z", ""))
        except (ValueError, TypeError):
            return 0
        return int(np.argmin(np.abs(self._time_dt - target)))

    # -- slicing ---------------------------------------------------------
    def slice(self, variable: str, depth: float, time: str | None = None) -> FieldSlice:
        """One depth/time plane of ``variable`` as a coarsened lat×lon grid.

        Reads only the strided plane from disk (never the full volume). NaN cells
        (land / analysis gaps) become ``None`` so the payload is valid JSON.
        """
        hvar = self._hvar(variable)
        units = _VARIABLES[variable][1]
        di = self._depth_index(depth)
        ti = self._time_index(time)

        plane = self._ds[hvar].isel(
            TIME=ti, DEPTH=di,
            LAT=slice(None, None, self._stride),
            LON=slice(None, None, self._stride),
        )
        arr = np.asarray(plane.values, dtype="float64")  # reads just this plane
        values = [[FieldSlice.clean(v) for v in row] for row in arr.tolist()]

        finite = arr[np.isfinite(arr)]
        vmin = round(float(finite.min()), 3) if finite.size else None
        vmax = round(float(finite.max()), 3) if finite.size else None

        return FieldSlice(
            variable=variable,
            units=units,
            depth=float(self._depth[di]),
            depth_requested=float(depth),
            time=self._time_iso[ti],
            time_index=ti,
            lat=list(self._lat),
            lon=list(self._lon),
            values=values,
            value_min=vmin,
            value_max=vmax,
        )

    def domain(self, variable: str) -> dict[str, float]:
        """Representative absolute range for a fixed colour scale.

        Sampled from the shallowest and deepest levels at the first timestep
        (two cheap coarsened reads) and cached — enough to colour every slice on
        one scale without scanning the full 10 GB volume.
        """
        if variable not in _VARIABLES:
            raise KeyError(variable)
        if variable not in self._domain_cache:
            vmins: list[float] = []
            vmaxs: list[float] = []
            for di in {0, len(self._depth) - 1}:
                sl = self.slice(variable, depth=self._depth[di], time=self._time_iso[0])
                if sl.value_min is not None:
                    vmins.append(sl.value_min)
                if sl.value_max is not None:
                    vmaxs.append(sl.value_max)
            self._domain_cache[variable] = {
                "min": min(vmins) if vmins else 0.0,
                "max": max(vmaxs) if vmaxs else 1.0,
            }
        return self._domain_cache[variable]
