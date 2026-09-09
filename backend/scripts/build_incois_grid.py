"""Preprocess the INCOIS objectively-analysed Argo CSV into a compact NetCDF.

The raw ERDDAP export (``incois_argo_mnt_McCreary_*.csv``) is ~900 MB / 10.2 M
rows — a fully-expanded (time, depth, lat, lon) table repeating every coordinate
on every row. This script reshapes the two fields the viewer needs
(``T_ANALYZED`` -> temperature, ``S_ANALYZED`` -> salinity) back into dense 4-D
arrays and writes a zlib-compressed ``incois_grid.nc`` (tens of MB) that the
backend opens lazily and slices per depth/time.

Grid (from the file): Indian Ocean, 1 deg, lat -29.5..29.5 (60), lon 30.5..119.5
(90), 24 depth levels 5..2000 m, 79 monthly steps 2020-01 .. 2026-07.

Run once (offline is fine — it only reads a local CSV and writes a local .nc):

    backend/.venv/bin/python backend/scripts/build_incois_grid.py \
        [CSV_PATH] [OUT_NC]
"""

from __future__ import annotations

import os
import sys
import time as _time

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_CSV = os.path.expanduser(
    "~/Downloads/incois_argo_mnt_McCreary_28b0_a27f_829f.csv"
)
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.abspath(os.path.join(_HERE, "..", "data", "incois_grid.nc"))

# Coordinate axes (verified against the file).
LATS = np.round(np.arange(-29.5, 29.5 + 0.5, 1.0), 1)     # 60
LONS = np.round(np.arange(30.5, 119.5 + 0.5, 1.0), 1)     # 90
DEPTHS = np.array(
    [5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500,
     600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000],
    dtype="float64",
)                                                          # 24


def _monthly_times() -> list[str]:
    """Monthly-on-the-15th ISO stamps, 2020-01 .. 2026-07 inclusive (79)."""
    out: list[str] = []
    y, m = 2020, 1
    while (y < 2026) or (y == 2026 and m <= 7):
        out.append(f"{y:04d}-{m:02d}-15T00:00:00Z")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def main() -> None:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    out_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT

    times = _monthly_times()
    assert len(times) == 79, f"expected 79 months, generated {len(times)}"
    time_ix = {s: i for i, s in enumerate(times)}

    nt, nz, nla, nlo = len(times), DEPTHS.size, LATS.size, LONS.size
    temp = np.full((nt, nz, nla, nlo), np.nan, dtype="float32")
    salt = np.full((nt, nz, nla, nlo), np.nan, dtype="float32")

    cols = ["time", "ZAX", "latitude", "longitude", "S_ANALYZED", "T_ANALYZED"]
    reader = pd.read_csv(
        csv_path,
        skiprows=[1],                       # drop the units row
        usecols=cols,
        dtype={
            "ZAX": "float64", "latitude": "float64", "longitude": "float64",
            "S_ANALYZED": "float32", "T_ANALYZED": "float32",
        },
        chunksize=1_000_000,
    )

    t0 = _time.time()
    rows = 0
    unmapped: set[str] = set()
    for ci, chunk in enumerate(reader):
        ti = chunk["time"].map(time_ix)
        bad = ti.isna()
        if bad.any():
            unmapped.update(chunk.loc[bad, "time"].unique().tolist())
            chunk = chunk.loc[~bad]
            ti = ti.loc[~bad]
        ti = ti.to_numpy(dtype="int64")

        lat = chunk["latitude"].to_numpy()
        lon = chunk["longitude"].to_numpy()
        zax = chunk["ZAX"].to_numpy()
        lai = np.rint(lat + 29.5).astype("int64")          # -29.5 -> 0
        loi = np.rint(lon - 30.5).astype("int64")          # 30.5  -> 0
        zi = np.searchsorted(DEPTHS, zax)                  # exact-match levels

        assert lai.min() >= 0 and lai.max() < nla, "lat index out of range"
        assert loi.min() >= 0 and loi.max() < nlo, "lon index out of range"
        assert zi.min() >= 0 and zi.max() < nz, "depth index out of range"

        temp[ti, zi, lai, loi] = chunk["T_ANALYZED"].to_numpy()
        salt[ti, zi, lai, loi] = chunk["S_ANALYZED"].to_numpy()

        rows += len(chunk)
        print(f"  chunk {ci:>2}  rows={rows:>10,}  {_time.time()-t0:5.1f}s",
              flush=True)

    if unmapped:
        raise SystemExit(f"unmapped time values in CSV: {sorted(unmapped)[:5]}")

    ds = xr.Dataset(
        {
            "temperature": (("time", "depth", "lat", "lon"), temp,
                            {"units": "degree_C", "long_name":
                             "Objectively analysed temperature (Argo)"}),
            "salinity": (("time", "depth", "lat", "lon"), salt,
                         {"units": "PSU", "long_name":
                          "Objectively analysed salinity (Argo)"}),
        },
        coords={
            # tz-naive UTC datetime64[ns]; the trailing "Z" would otherwise make
            # pandas hand xarray a tz-aware dtype the CF encoder can't write.
            "time": pd.to_datetime(times, utc=True).tz_convert(None).to_numpy(),
            "depth": DEPTHS,
            "lat": LATS,
            "lon": LONS,
        },
        attrs={
            "title": "INCOIS objectively-analysed monthly Argo grid (Indian Ocean)",
            "source_csv": os.path.basename(csv_path),
            "region": "Indian Ocean 30.5E-119.5E, 29.5S-29.5N, 1 deg",
        },
    )
    enc = {v: {"zlib": True, "complevel": 5, "dtype": "float32", "_FillValue": np.float32(np.nan)}
           for v in ("temperature", "salinity")}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds.to_netcdf(out_path, encoding=enc)

    size_mb = os.path.getsize(out_path) / 1e6
    tvalid = int(np.isfinite(temp).sum())
    svalid = int(np.isfinite(salt).sum())
    print(f"\nwrote {out_path}  ({size_mb:.1f} MB)")
    print(f"  temperature valid cells: {tvalid:,}  "
          f"range {np.nanmin(temp):.3f}..{np.nanmax(temp):.3f} degC")
    print(f"  salinity    valid cells: {svalid:,}  "
          f"range {np.nanmin(salt):.3f}..{np.nanmax(salt):.3f} PSU")
    print(f"  surface (5 m) temp, first month: "
          f"{np.nanmin(temp[0,0]):.2f}..{np.nanmax(temp[0,0]):.2f} degC")


if __name__ == "__main__":
    main()
