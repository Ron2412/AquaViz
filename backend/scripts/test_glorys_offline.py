"""Offline smoke test for the GLORYS pipeline — no network, no real data.

Fabricates CMEMS-shaped raw NetCDF fixtures in $TMPDIR, runs
build_glorys_grid.py on them, and reads the result back through
GlorysGriddedSource. Deliberately does NOT touch backend/app/data/ so the live
INCOIS view stays active.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
import xarray as xr

REPO = "/Users/ronakdas/AquaViz/backend"
PY = os.path.join(REPO, ".venv", "bin", "python")
BUILD = os.path.join(REPO, "scripts", "build_glorys_grid.py")
sys.path.insert(0, REPO)

GLOBAL_LAT = np.round(np.arange(-89.5, 90.0, 1.0), 1)     # 180
GLOBAL_LON = np.round(np.arange(-179.5, 180.0, 1.0), 1)   # 360


def _build(raw: str, out: str) -> None:
    r = subprocess.run([PY, BUILD, raw, out], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"build failed:\nSTDOUT\n{r.stdout}\nSTDERR\n{r.stderr}")
    print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "(built)")


def _mk(path, *, lat, lon, depth, times, with_currents, continent=None):
    """Write a raw CMEMS-style file. `continent` masks a lat/lon box to NaN."""
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    base = 30.0 * np.cos(np.deg2rad(LAT))           # warm equator, cold poles
    if continent is not None:
        la0, la1, lo0, lo1 = continent
        mask = (LAT >= la0) & (LAT <= la1) & (LON >= lo0) & (LON <= lo1)
    else:
        mask = np.zeros_like(base, dtype=bool)
    base = np.where(mask, np.nan, base)

    def stack(field2d):
        a = np.broadcast_to(field2d, (times.size, depth.size, *field2d.shape))
        return np.array(a, dtype="float32")

    sal = np.where(mask, np.nan, 35.0)
    dv = {
        "thetao": (("time", "depth", "latitude", "longitude"), stack(base)),
        "so": (("time", "depth", "latitude", "longitude"), stack(sal)),
    }
    if with_currents:
        uo = np.where(mask, np.nan, 0.3)
        vo = np.where(mask, np.nan, 0.4)              # |(.3,.4)| = 0.5
        dv["uo"] = (("time", "depth", "latitude", "longitude"), stack(uo))
        dv["vo"] = (("time", "depth", "latitude", "longitude"), stack(vo))
    ds = xr.Dataset(dv, coords={"time": times, "depth": depth,
                                "latitude": lat, "longitude": lon})
    ds.to_netcdf(path)


def test_native_1deg_with_currents(tmp):
    raw, out = os.path.join(tmp, "t1_raw.nc"), os.path.join(tmp, "t1_grid.nc")
    depth = np.array([0.5, 10, 55, 150, 500, 1000, 2000], dtype="float64")
    times = np.array(["2020-01-15", "2020-02-15"], dtype="datetime64[ns]")
    _mk(raw, lat=GLOBAL_LAT, lon=GLOBAL_LON, depth=depth, times=times,
        with_currents=True, continent=(9.5, 20.5, 9.5, 20.5))
    _build(raw, out)

    from app.data.glorys import GlorysGriddedSource
    src = GlorysGriddedSource(out)

    assert src.variables() == ["temperature", "salinity", "current_speed"], src.variables()
    assert src.depths() == [0.5, 10.0, 55.0, 150.0, 500.0, 1000.0, 2000.0], src.depths()
    assert len(src.times()) == 2, src.times()

    sl = src.slice("temperature", depth=0.5, time=None)
    assert len(sl.values) == 180 and len(sl.values[0]) == 360, "grid shape"
    # equator ocean (lat 0.5, lon 0.5) finite; continent (lat 15.5, lon 15.5) None
    assert sl.values[90][180] is not None and sl.values[90][180] > 20, sl.values[90][180]
    assert sl.values[105][195] is None, sl.values[105][195]

    cs = src.slice("current_speed", depth=10, time=None)
    assert abs(cs.values[90][180] - 0.5) < 1e-4, cs.values[90][180]
    assert cs.values[105][195] is None, "current_speed masked on land"
    dom = src.domain("current_speed")
    assert dom["min"] >= 0.0 and abs(dom["max"] - 0.5) < 1e-3, dom
    print("T1 native-1° + currents: PASS")


def test_half_deg_coarsen_and_lon_wrap(tmp):
    raw, out = os.path.join(tmp, "t2_raw.nc"), os.path.join(tmp, "t2_grid.nc")
    lat = np.arange(-89.75, 90.0, 0.5)                 # 0.5° -> coarsen 2
    lon = np.arange(0.0, 360.0, 0.5)                   # 0..359.5 -> normalised
    depth = np.array([1.0, 100.0], dtype="float64")
    times = np.array(["2020-06-15"], dtype="datetime64[ns]")
    _mk(raw, lat=lat, lon=lon, depth=depth, times=times, with_currents=False)
    _build(raw, out)

    from app.data.glorys import GlorysGriddedSource
    src = GlorysGriddedSource(out)
    assert src.variables() == ["temperature", "salinity"], src.variables()
    assert src.depths() == [1.0, 100.0], src.depths()

    sl = src.slice("temperature", depth=1.0, time=None)
    assert len(sl.values) == 180 and len(sl.values[0]) == 360, "grid shape"
    finite = sum(1 for row in sl.values for v in row if v is not None)
    frac = finite / (180 * 360)
    assert frac > 0.5, f"expected mostly-finite after coarsen+wrap, got {frac:.2f}"
    print(f"T2 0.5° coarsen + lon-wrap: PASS (finite {frac:.0%})")


def test_no_currents(tmp):
    raw, out = os.path.join(tmp, "t3_raw.nc"), os.path.join(tmp, "t3_grid.nc")
    depth = np.array([5.0, 500.0], dtype="float64")
    times = np.array(["2020-03-15"], dtype="datetime64[ns]")
    _mk(raw, lat=GLOBAL_LAT, lon=GLOBAL_LON, depth=depth, times=times,
        with_currents=False)
    _build(raw, out)

    from app.data.glorys import GlorysGriddedSource
    src = GlorysGriddedSource(out)
    assert src.variables() == ["temperature", "salinity"], src.variables()
    for call in (lambda: src.slice("current_speed", depth=5, time=None),
                 lambda: src.units("current_speed")):
        try:
            call()
        except KeyError:
            pass
        else:
            raise AssertionError("expected KeyError for current_speed without uo/vo")
    print("T3 no-currents: PASS")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        test_native_1deg_with_currents(tmp)
        test_half_deg_coarsen_and_lon_wrap(tmp)
        test_no_currents(tmp)
    print("\nall GLORYS tests passed")
