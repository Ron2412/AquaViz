"""Preprocess a Copernicus GLORYS download into the compact global grid AquaViz serves.

Source: **GLOBAL_MULTIYEAR_PHY_001_030** (GLORYS12V1), native 1/12° (~8 km), 50
depth levels — far too dense to ship to the browser per slice. This script
regrids a downloaded subset onto the viewer's global **1°** mesh (180×360) and
writes a zlib-compressed ``glorys_grid.nc`` (tens of MB) that
:class:`app.data.glorys.GlorysGriddedSource` opens lazily. It keeps four fields
— ``thetao``→temperature, ``so``→salinity, and the current components
``uo``/``vo`` (the source derives ``current_speed`` from them and keeps the
components for a future arrow layer).

Regridding is deliberately SciPy-free (the backend venv is NumPy/xarray only):

* **Horizontal** — coarsen-average 1/12°→1° with :func:`numpy.nanmean` (so land
  NaNs don't bleed into a coastal cell's average), then snap to the canonical 1°
  centres with a nearest-index ``.sel`` and mask anything outside the download's
  own lat/lon span (a bbox subset shows data only where it exists).
* **Vertical** — keep GLORYS's own levels down to ``DEPTH_CAP`` m rather than
  interpolate; the viewer snaps the depth slider to the nearest available level.

Download the subset first with the Copernicus Marine toolbox (free account):

    pip install copernicusmarine        # once
    copernicusmarine subset \
        --dataset-id cmems_mod_glo_phy_my_0.083deg_P1M-m \
        --variable thetao --variable so --variable uo --variable vo \
        --start-datetime 2020-01-01T00:00:00 --end-datetime 2021-12-31T00:00:00 \
        --minimum-depth 0 --maximum-depth 2000 \
        -o backend/data -f glorys_raw.nc

Subsetting by time/depth (and optionally a bbox) keeps the download sane — a
full global 1/12° archive is enormous. Then build the compact grid (writes to
``backend/data/glorys_grid.nc``, the dir the source factory scans):

    backend/.venv/bin/python backend/scripts/build_glorys_grid.py \
        [RAW_NC] [OUT_NC]
"""

from __future__ import annotations

import os
import sys
import time as _time
import warnings

import numpy as np
import xarray as xr

_HERE = os.path.dirname(os.path.abspath(__file__))
# settings.data_dir is backend/data/ — the dir the source factory scans. The
# output MUST land here or _find_glorys_file() won't see it (same place the
# INCOIS grid and Argo *_prof.nc files live).
DEFAULT_RAW = os.path.abspath(os.path.join(_HERE, "..", "data", "glorys_raw.nc"))
DEFAULT_OUT = os.path.abspath(os.path.join(_HERE, "..", "data", "glorys_grid.nc"))

# Canonical global 1° mesh (identical to app/data/glorys.py and incois.py).
GLOBAL_LAT = np.round(np.arange(-89.5, 90.0, 1.0), 1)      # 180
GLOBAL_LON = np.round(np.arange(-179.5, 180.0, 1.0), 1)    # 360

# Keep GLORYS's own levels shallower than this (m); caps file size, matches the
# viewer's Argo/INCOIS depth range. Raise if you need the abyssal ocean.
DEPTH_CAP = 2000.0

# CMEMS variable names, with fallbacks, mapped to our output names.
VAR_ALIASES = {
    "temperature": ["thetao", "temperature", "to", "TEMP"],
    "salinity": ["so", "salinity", "PSAL"],
    "uo": ["uo", "u", "utotal", "eastward_sea_water_velocity"],
    "vo": ["vo", "v", "vtotal", "northward_sea_water_velocity"],
}
_COORD_ALIASES = {
    "latitude": ["latitude", "lat", "nav_lat", "y"],
    "longitude": ["longitude", "lon", "nav_lon", "x"],
    "depth": ["depth", "deptht", "lev", "z"],
    "time": ["time", "time_counter"],
}


def _first_present(ds: xr.Dataset, names: list[str]) -> str | None:
    for n in names:
        if n in ds.variables or n in ds.coords or n in ds.dims:
            return n
    return None


def _standardise_coords(ds: xr.Dataset) -> xr.Dataset:
    """Rename coords to latitude/longitude/depth/time; normalise lon to [-180,180)."""
    rename = {}
    for target, aliases in _COORD_ALIASES.items():
        found = _first_present(ds, aliases)
        if found and found != target:
            rename[found] = target
    if rename:
        ds = ds.rename(rename)
    for req in ("latitude", "longitude", "depth", "time"):
        if req not in ds.coords and req not in ds.dims:
            raise SystemExit(f"input is missing a '{req}' coordinate (have: {list(ds.coords)})")
    # Longitudes as [-180,180) so they register onto GLOBAL_LON.
    lon = ds["longitude"]
    if float(lon.max()) > 180.0:
        ds = ds.assign_coords(longitude=(((lon + 180) % 360) - 180)).sortby("longitude")
    return ds.sortby("latitude").sortby("depth")


def _coarsen_factor(coord: np.ndarray) -> int:
    """How many native cells span ~1°, from the median coordinate spacing."""
    if coord.size < 2:
        return 1
    step = float(np.median(np.abs(np.diff(coord))))
    return max(1, int(round(1.0 / step))) if step > 0 else 1


def main() -> None:
    raw_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RAW
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    if not os.path.isfile(raw_path):
        raise SystemExit(f"raw GLORYS file not found: {raw_path}\n"
                         "(see this script's docstring for the copernicusmarine command)")

    ds = _standardise_coords(xr.open_dataset(raw_path))

    # Cap depth to keep the file small; keep GLORYS's own levels within range.
    ds = ds.sel(depth=slice(0.0, DEPTH_CAP))
    if ds.sizes.get("depth", 0) == 0:
        raise SystemExit(f"no depth levels within 0..{DEPTH_CAP} m in input")

    # Resolve which fields are present. Temperature + salinity are required;
    # currents are optional (some subsets omit uo/vo).
    resolved: dict[str, str] = {}
    for out_name, aliases in VAR_ALIASES.items():
        src = _first_present(ds, aliases)
        if src is not None:
            resolved[out_name] = src
    for req in ("temperature", "salinity"):
        if req not in resolved:
            raise SystemExit(f"input has no {req} variable (looked for {VAR_ALIASES[req]})")
    has_currents = "uo" in resolved and "vo" in resolved

    times = np.asarray(ds["time"].values)
    out_depths = np.asarray(ds["depth"].values, dtype="float64")
    src_lat = np.asarray(ds["latitude"].values, dtype="float64")
    src_lon = np.asarray(ds["longitude"].values, dtype="float64")
    fac_lat = _coarsen_factor(src_lat)
    fac_lon = _coarsen_factor(src_lon)

    # Target cells outside the download's own span stay empty (no edge smear).
    lat_oob = (GLOBAL_LAT < src_lat.min()) | (GLOBAL_LAT > src_lat.max())
    lon_oob = (GLOBAL_LON < src_lon.min()) | (GLOBAL_LON > src_lon.max())

    nt, nz = times.size, out_depths.size
    out: dict[str, np.ndarray] = {
        name: np.full((nt, nz, GLOBAL_LAT.size, GLOBAL_LON.size), np.nan, dtype="float32")
        for name in resolved
    }

    def _regrid(da: xr.DataArray) -> np.ndarray:
        """One (depth, lat, lon) stack -> the canonical (nz, 180, 360) grid."""
        if fac_lat > 1 or fac_lon > 1:
            # NaN-aware block average so land doesn't poison a coastal cell.
            da = da.coarsen(latitude=fac_lat, longitude=fac_lon,
                            boundary="trim").reduce(np.nanmean)
        da = da.sel(latitude=GLOBAL_LAT, longitude=GLOBAL_LON, method="nearest")
        arr = np.asarray(da.values, dtype="float32")
        arr[..., lat_oob, :] = np.nan
        arr[..., :, lon_oob] = np.nan
        return arr

    t0 = _time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN windows
        for ti in range(nt):
            for out_name, src in resolved.items():
                out[out_name][ti] = _regrid(ds[src].isel(time=ti))
            print(f"  time {ti + 1:>3}/{nt}  {_time.time() - t0:5.1f}s", flush=True)

    data_vars = {
        "temperature": (("time", "depth", "lat", "lon"), out["temperature"],
                        {"units": "degree_C", "long_name": "Sea water potential temperature (GLORYS)"}),
        "salinity": (("time", "depth", "lat", "lon"), out["salinity"],
                     {"units": "PSU", "long_name": "Sea water salinity (GLORYS)"}),
    }
    if has_currents:
        data_vars["uo"] = (("time", "depth", "lat", "lon"), out["uo"],
                           {"units": "m s-1", "long_name": "Eastward sea water velocity (GLORYS)"})
        data_vars["vo"] = (("time", "depth", "lat", "lon"), out["vo"],
                           {"units": "m s-1", "long_name": "Northward sea water velocity (GLORYS)"})

    result = xr.Dataset(
        data_vars,
        coords={"time": times, "depth": out_depths,
                "lat": GLOBAL_LAT, "lon": GLOBAL_LON},
        attrs={
            "title": "Copernicus GLORYS12V1 regridded to global 1° (AquaViz)",
            "source_product": "GLOBAL_MULTIYEAR_PHY_001_030",
            "source_file": os.path.basename(raw_path),
            "note": "current_speed is derived at read time as hypot(uo, vo)",
        },
    )
    enc = {v: {"zlib": True, "complevel": 5, "dtype": "float32",
               "_FillValue": np.float32(np.nan)} for v in data_vars}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_netcdf(out_path, encoding=enc)
    ds.close()

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\nwrote {out_path}  ({size_mb:.1f} MB)")
    print(f"  grid: {nt} times × {nz} depths ({out_depths.min():.1f}..{out_depths.max():.1f} m)"
          f" × {GLOBAL_LAT.size} lat × {GLOBAL_LON.size} lon  (coarsen {fac_lat}×{fac_lon})")
    for name in ("temperature", "salinity"):
        a = out[name]
        print(f"  {name:<12} valid {int(np.isfinite(a).sum()):>10,}  "
              f"range {np.nanmin(a):.3f}..{np.nanmax(a):.3f}")
    if has_currents:
        spd = np.hypot(out["uo"], out["vo"])
        print(f"  current_speed valid {int(np.isfinite(spd).sum()):>10,}  "
              f"range {np.nanmin(spd):.3f}..{np.nanmax(spd):.3f} m/s")
    else:
        print("  (no uo/vo in input — current_speed will be unavailable)")


if __name__ == "__main__":
    main()
