"""Data-source factory.

Returns the active :class:`OceanDataSource`. Today that is always the synthetic
NumPy source (works offline, no dependencies beyond NumPy). Once xarray/netCDF4
are installed and a ``.nc`` file is present in ``settings.data_dir``, this is
where a NetCDF-backed source will be selected instead — routers won't change.
"""

from __future__ import annotations

import glob
import os

from app.config import settings

from .base import FieldSlice, OceanDataSource
from .instruments import InstrumentSource
from .synthetic import SyntheticOceanSource

__all__ = [
    "FieldSlice",
    "OceanDataSource",
    "get_source",
    "get_instruments",
]

_source: OceanDataSource | None = None
_instruments: object | None = None
_argo: object | None = None
_argo_loaded = False


def _find_argo_file() -> str | None:
    """First Argo profile NetCDF (``*_prof.nc``) in the data dir, if any."""
    if not os.path.isdir(settings.data_dir):
        return None
    matches = sorted(glob.glob(os.path.join(settings.data_dir, "*_prof.nc")))
    return matches[0] if matches else None


def _find_incois_file() -> str | None:
    """Compact INCOIS gridded product, if built (scripts/build_incois_grid.py)."""
    path = os.path.join(settings.data_dir, "incois_grid.nc")
    return path if os.path.isfile(path) else None


def _find_glorys_file() -> str | None:
    """Compact global GLORYS grid, if built (scripts/build_glorys_grid.py)."""
    path = os.path.join(settings.data_dir, "glorys_grid.nc")
    return path if os.path.isfile(path) else None


def _get_argo():
    """Load the Argo source once (shared by field blend + instruments)."""
    global _argo, _argo_loaded
    if not _argo_loaded:
        _argo_loaded = True
        path = _find_argo_file()
        if path is not None:
            try:
                from .argo import ArgoInstrumentSource

                _argo = ArgoInstrumentSource(path)
            except Exception:
                # Unreadable/malformed file: degrade gracefully to synthetic.
                _argo = None
    return _argo


def get_source() -> OceanDataSource:
    """Return the process-wide field source (constructed once, then cached).

    Preference order, richest first:

    1. **GLORYS** (``glorys_grid.nc``) — global temperature/salinity **and
       currents**; strictly the widest coverage, so it wins when present.
    2. **INCOIS** (``incois_grid.nc``) — real objectively-analysed Argo
       temperature/salinity over the Indian Ocean.
    3. **Synthetic** global field — nudged toward Argo observations near each
       float (:class:`BlendedOceanSource`) when a profile file is present, else
       plain synthetic.

    All expose the same surface, so routers don't change.
    """
    global _source
    if _source is None:
        # 1. Prefer the global GLORYS grid when present (adds currents).
        glorys = _find_glorys_file()
        if glorys is not None:
            try:
                from .glorys import GlorysGriddedSource

                _source = GlorysGriddedSource(glorys)
                return _source
            except Exception:
                # Unreadable/malformed file: fall through to the next option.
                pass
        # 2. Otherwise the real INCOIS gridded product.
        incois = _find_incois_file()
        if incois is not None:
            try:
                from .incois import IncoisGriddedSource

                _source = IncoisGriddedSource(incois)
                return _source
            except Exception:
                # Unreadable/malformed file: degrade to synthetic/blended.
                pass
        synth = SyntheticOceanSource()
        argo = _get_argo()
        if argo is not None:
            try:
                from .blended import BlendedOceanSource

                _source = BlendedOceanSource(synth, argo)
            except Exception:
                _source = synth
        else:
            _source = synth
    return _source


def get_instruments():
    """Return the process-wide instrument source.

    Prefers a real Argo ``*_prof.nc`` file in ``settings.data_dir`` when present
    (real float markers + measured profiles); otherwise falls back to the
    synthetic roster sampled from the model field. Both expose the same surface.
    """
    global _instruments
    if _instruments is None:
        argo = _get_argo()
        if argo is not None:
            _instruments = argo
        else:
            # Profiles are sampled from the same field model, so pass it in.
            base = SyntheticOceanSource()
            _instruments = InstrumentSource(base)
    return _instruments
