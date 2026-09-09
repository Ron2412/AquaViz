"""Real Argo float instruments loaded from a NetCDF multi-profile file.

Reads a standard Argo GDAC ``*_prof.nc`` file (Argo-3.1 / CF-1.6) and exposes
the *same* surface as :class:`~app.data.instruments.InstrumentSource` so the
routers and frontend don't change: a roster of clickable markers plus a
depth-vs-value profile per platform.

Each ``N_PROF`` row is one float cycle at a fixed ``(LATITUDE, LONGITUDE)`` with
paired ``PRES`` / ``TEMP`` / ``PSAL`` arrays over ``N_LEVELS``. We:

* keep only good samples (QC flag in ``{"1", "2"}``) — this drops the obvious
  junk (e.g. ``PSAL`` spikes of 0 or 61 PSU seen in the raw file),
* treat pressure in dbar as depth in metres (within ~2% in the ocean; good
  enough for a depth axis and avoids a TEOS-10 dependency),
* load everything once at construction (a single file is small: tens of floats,
  ~1k levels each) and serve slices from memory.

Swap the file behind the same class and nothing downstream cares.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Only samples with these QC flags are trusted: 1 = good, 2 = probably good.
_GOOD_QC = {"1", "2"}

# Argo pressure is in dbar; 1 dbar ≈ 1.0 m of depth in seawater to within ~2%.
_DBAR_TO_M = 1.0

# Variable -> (NetCDF var name, units) for the two core Argo parameters.
_VARIABLES = {
    "temperature": ("TEMP", "degree_C"),
    "salinity": ("PSAL", "PSU"),
}


def _decode(value: object) -> str:
    """Argo stores fixed-width strings as bytes; normalise to a clean str."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace").strip()
    return str(value).strip()


@dataclass(frozen=True)
class ArgoInstrument:
    """One Argo profile (a float cycle) presented as a deployed instrument."""

    id: str
    kind: str
    name: str
    lat: float
    lon: float
    variables: tuple[str, ...]
    time: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "variables": list(self.variables),
            "time": self.time,
        }


class ArgoInstrumentSource:
    """Roster + profiles read from a real Argo ``*_prof.nc`` file."""

    name = "argo-netcdf"

    def __init__(self, path: str) -> None:
        self._path = path
        self._instruments: list[ArgoInstrument] = []
        self._by_id: dict[str, ArgoInstrument] = {}
        # id -> {variable -> (depths[], values[])}, prebuilt at load time.
        self._profiles: dict[str, dict[str, tuple[list[float], list[float]]]] = {}
        self._load()

    # -- loading -------------------------------------------------------------
    def _load(self) -> None:
        import numpy as np
        import xarray as xr

        ds = xr.open_dataset(self._path)
        try:
            lat = ds.LATITUDE.values
            lon = ds.LONGITUDE.values
            juld = ds.JULD.values
            platform = [_decode(p) for p in ds.PLATFORM_NUMBER.values]
            pres = ds.PRES.values  # (N_PROF, N_LEVELS), dbar
            n_prof = lat.shape[0]

            # Per-variable value + QC arrays, fetched once.
            var_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for key, (nc_name, _units) in _VARIABLES.items():
                qc_name = f"{nc_name}_QC"
                if nc_name in ds and qc_name in ds:
                    var_arrays[key] = (ds[nc_name].values, ds[qc_name].values)

            pres_qc = ds["PRES_QC"].values if "PRES_QC" in ds else None
        finally:
            ds.close()

        for i in range(n_prof):
            la = float(lat[i])
            lo = float(lon[i])
            if not np.isfinite(la) or not np.isfinite(lo):
                continue

            plat = platform[i] or f"prof{i}"
            inst_id = f"argo-{plat}"
            # Guard against duplicate platform numbers within one file.
            if inst_id in self._by_id:
                inst_id = f"{inst_id}-{i}"

            # np.datetime64 NaT stringifies as "NaT"; treat that as unknown time.
            time_str = "" if str(juld[i]) == "NaT" else str(juld[i])[:19]

            depth_col = pres[i] * _DBAR_TO_M
            pres_flags = (
                [_decode(x) for x in pres_qc[i]] if pres_qc is not None else None
            )

            available: list[str] = []
            profiles_for_inst: dict[str, tuple[list[float], list[float]]] = {}
            for key, (values_arr, qc_arr) in var_arrays.items():
                col = values_arr[i]
                flags = [_decode(x) for x in qc_arr[i]]
                depths: list[float] = []
                vals: list[float] = []
                for level in range(col.shape[0]):
                    v = col[level]
                    d = depth_col[level]
                    if not np.isfinite(v) or not np.isfinite(d):
                        continue
                    if flags[level] not in _GOOD_QC:
                        continue
                    if pres_flags is not None and pres_flags[level] not in _GOOD_QC:
                        continue
                    depths.append(round(float(d), 1))
                    vals.append(round(float(v), 3))
                if depths:
                    profiles_for_inst[key] = (depths, vals)
                    available.append(key)

            if not available:
                continue  # no good data for this float; skip the marker

            inst = ArgoInstrument(
                id=inst_id,
                kind="argo",
                name=f"Argo {plat}",
                lat=round(la, 4),
                lon=round(lo, 4),
                variables=tuple(available),
                time=time_str,
            )
            self._instruments.append(inst)
            self._by_id[inst_id] = inst
            self._profiles[inst_id] = profiles_for_inst

    # -- catalogue -----------------------------------------------------------
    def list(
        self, bbox: tuple[float, float, float, float] | None = None
    ) -> list[ArgoInstrument]:
        """All floats, optionally filtered to ``(min_lon,min_lat,max_lon,max_lat)``."""
        if bbox is None:
            return list(self._instruments)
        min_lon, min_lat, max_lon, max_lat = bbox
        return [
            inst
            for inst in self._instruments
            if min_lon <= inst.lon <= max_lon and min_lat <= inst.lat <= max_lat
        ]

    def get(self, instrument_id: str) -> ArgoInstrument:
        return self._by_id[instrument_id]

    def profiles_for(self, variable: str):
        """Every float's measured profile of ``variable`` as arrays.

        Returns a list of ``(lat, lon, depths, values)`` where ``depths`` and
        ``values`` are 1-D NumPy arrays sorted by increasing depth. Used by the
        blended field source to nudge the synthetic grid toward observations.
        """
        import numpy as np

        out: list[tuple[float, float, "np.ndarray", "np.ndarray"]] = []
        for inst in self._instruments:
            by_var = self._profiles.get(inst.id, {})
            pv = by_var.get(variable)
            if not pv:
                continue
            depths, values = pv
            out.append(
                (inst.lat, inst.lon, np.asarray(depths, dtype=float),
                 np.asarray(values, dtype=float))
            )
        return out

    # -- profiles ------------------------------------------------------------
    def profile(
        self, instrument_id: str, variable: str, time: str | None = None
    ) -> dict:
        """Measured depth profile of ``variable`` for one float.

        ``time`` is ignored: a float cycle has a single fixed observation time
        (returned in the payload). The parameter is kept for signature parity
        with the synthetic source, which can be scrubbed through model timesteps.
        """
        inst = self._by_id.get(instrument_id)
        if inst is None:
            raise KeyError(instrument_id)
        by_var = self._profiles.get(instrument_id, {})
        if variable not in by_var:
            raise KeyError(variable)

        depths, values = by_var[variable]
        units = _VARIABLES[variable][1]
        return {
            "id": inst.id,
            "kind": inst.kind,
            "name": inst.name,
            "lat": inst.lat,
            "lon": inst.lon,
            "variable": variable,
            "units": units,
            "time": inst.time,
            "depths": depths,
            "values": values,
        }
