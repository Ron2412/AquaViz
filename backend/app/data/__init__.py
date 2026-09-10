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
    "get_model",
    "get_comparison_model",
    "available_models",
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


def _find_hycom_file() -> str | None:
    """The large HYCOM model NetCDF.

    ``MODEL_DATA_PATH`` if it points at a real file, else any ``*hycom*.nc``
    (case-insensitive) in the data dir. The file is ~10 GB and never committed;
    a symlink into the data dir typically points at wherever it lives.
    """
    if settings.model_data_path and os.path.isfile(settings.model_data_path):
        return settings.model_data_path
    if os.path.isdir(settings.data_dir):
        matches = sorted(
            p for p in glob.glob(os.path.join(settings.data_dir, "*.nc"))
            if "hycom" in os.path.basename(p).lower()
        )
        if matches:
            return matches[0]
    return None


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

    1. **HYCOM** (``MODEL_DATA_PATH`` or ``*hycom*.nc``) — a genuine independent
       ocean model, read lazily from its ~10 GB NetCDF. Wins when present.
    2. **GLORYS** (``glorys_grid.nc``) — global temperature/salinity **and
       currents**.
    3. **INCOIS** (``incois_grid.nc``) — real objectively-analysed Argo
       temperature/salinity over the Indian Ocean.
    4. **Synthetic** global field — nudged toward Argo observations near each
       float (:class:`BlendedOceanSource`) when a profile file is present, else
       plain synthetic.

    All expose the same surface, so routers don't change.
    """
    global _source
    if _source is None:
        # 0. Prefer a real HYCOM model file when present — a genuine independent
        #    ocean model read lazily from its ~10 GB NetCDF (never fully loaded).
        hycom = _find_hycom_file()
        if hycom is not None:
            try:
                from .hycom import HycomModelSource

                _source = HycomModelSource(hycom)
                return _source
            except Exception:
                # Unreadable/malformed file: fall through to the next option.
                pass
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


# --- Named model registry (for the comparison's optional ``model=`` selector) ---

_models: dict[str, object] = {}


def _build_model(key: str):
    """Construct a named model source (uncached); None if its file is absent."""
    if key == "hycom":
        path = _find_hycom_file()
        if path:
            from .hycom import HycomModelSource

            return HycomModelSource(path)
    elif key == "glorys":
        path = _find_glorys_file()
        if path:
            from .glorys import GlorysGriddedSource

            return GlorysGriddedSource(path)
    elif key == "incois":
        path = _find_incois_file()
        if path:
            from .incois import IncoisGriddedSource

            return IncoisGriddedSource(path)
    return None


def available_models() -> list[str]:
    """Model keys whose file is present on disk (candidates for ``model=``)."""
    keys = []
    if _find_hycom_file():
        keys.append("hycom")
    if _find_glorys_file():
        keys.append("glorys")
    if _find_incois_file():
        keys.append("incois")
    return keys


def get_model(key: str | None):
    """Return a named model source for the comparison (cached per key).

    ``key`` is ``"hycom"`` | ``"glorys"`` | ``"incois"``; ``None`` or ``"auto"``
    returns the primary field source (:func:`get_source`). Raises ``KeyError``
    for an unknown or unavailable model — the router surfaces that as 404.
    """
    if not key or key == "auto":
        return get_source()
    key = key.lower()
    if key not in _models:
        src = _build_model(key)
        if src is None:
            raise KeyError(key)
        _models[key] = src
    return _models[key]


def _parse_iso(value) -> "datetime | None":  # noqa: F821 - forward ref in annotation
    from datetime import datetime

    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(
            tzinfo=None
        )
    except (ValueError, TypeError):
        return None


def get_comparison_model(instruments=None):
    """Model the comparison should use by default: the richest one that overlaps
    the observations in time.

    A model-vs-observation matchup is only meaningful when the two overlap in
    time, so this returns the richest model whose timestep range brackets the
    loaded Argo observations (with a small pad). HYCOM here is a near-real-time
    forecast *week*; when the loaded Argo is historical it won't overlap, and the
    historical INCOIS analysis is chosen instead — the response labels whichever
    model was used. Falls back to :func:`get_source` when no observation times
    are available. A caller can still force a specific model via :func:`get_model`.
    """
    from datetime import timedelta

    primary = get_source()
    if instruments is None:
        return primary
    try:
        obs_times = sorted(
            f.time for f in instruments.list(None) if getattr(f, "time", None)
        )
    except Exception:
        return primary
    if not obs_times:
        return primary
    lo, hi = _parse_iso(obs_times[0]), _parse_iso(obs_times[-1])
    if lo is None or hi is None:
        return primary

    pad = timedelta(days=45)
    for key in ("hycom", "glorys", "incois"):
        try:
            src = get_model(key)
        except KeyError:
            continue
        model_times = [t for t in (_parse_iso(x) for x in src.times()) if t]
        if not model_times:
            continue
        if min(model_times) - pad <= hi and lo <= max(model_times) + pad:
            return src
    return primary
