"""Lazily inspect the HYCOM NetCDF structure before writing any reader.

Prints dimensions, coordinates (with ranges), and data variables (dims / dtype /
shape / units) WITHOUT loading any data variable into memory. Coordinates are 1-D
and small, so reading their min/max is safe; data variables are only described.

Usage:
    ./.venv/bin/python scripts/inspect_hycom.py /abs/path/to/hycom.nc
"""

import sys

import numpy as np
import xarray as xr


def main(path: str) -> None:
    # Lazy open: the netCDF4 backend reads headers/coords only. Without dask,
    # xarray still wraps data variables in lazy on-disk arrays — a single
    # depth/time slice reads just that slice, never the whole 10 GB.
    ds = xr.open_dataset(path, decode_times=True)
    print(f"file: {path}")
    print(f"engine: netCDF4 (lazy)  on-disk logical size: {ds.nbytes/1e9:.2f} GB\n")

    print("== dimensions ==")
    for name, size in ds.sizes.items():
        print(f"  {name}: {size}")

    print("\n== coordinates ==")
    for name, c in ds.coords.items():
        try:
            vals = np.asarray(c.values)
            if vals.size:
                lo, hi = vals.min(), vals.max()
                step = None
                if vals.ndim == 1 and vals.size > 1:
                    diffs = np.abs(np.diff(vals.astype("float64", copy=False))) \
                        if not np.issubdtype(vals.dtype, np.datetime64) else None
                    if diffs is not None:
                        step = float(np.median(diffs))
                rng = f"{lo} .. {hi}" + (f"  ~step {step:.4g}" if step else "")
            else:
                rng = "(empty)"
        except Exception as e:  # noqa: BLE001 - inspection tool, report and move on
            rng = f"(unreadable: {e})"
        units = c.attrs.get("units", "")
        print(f"  {name}  dims={c.dims} dtype={c.dtype} size={c.size}  {rng}  {units}")

    print("\n== data variables (NOT loaded) ==")
    for name, v in ds.data_vars.items():
        print(f"  {name}  dims={v.dims} shape={v.shape} dtype={v.dtype}")
        for a in ("standard_name", "long_name", "units", "_FillValue", "missing_value"):
            if a in v.attrs:
                print(f"      {a}: {v.attrs[a]}")

    print("\n== global attrs (subset) ==")
    for a in ("title", "institution", "source", "history", "Conventions",
              "experiment", "time_coverage_start", "time_coverage_end"):
        if a in ds.attrs:
            print(f"  {a}: {str(ds.attrs[a])[:120]}")

    ds.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
