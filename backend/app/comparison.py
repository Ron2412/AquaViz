"""Model-vs-observation comparison.

Answers the platform's core question: *what did the ocean model estimate, what
did the instrument actually observe, and how different are they?*

The **observations** are Argo float profiles (:class:`~app.data.argo.ArgoInstrumentSource`)
— real in-situ measurements at a fixed ``(lat, lon, time)``. The **model** is the
active gridded field source (:class:`~app.data.base.OceanDataSource`): the INCOIS
objective analysis today, the HYCOM grid once its file is present. Because both
speak the same ``slice()`` interface, this matcher never needs to know which one
it is comparing against.

Matching rules (nearest-neighbour, each dimension independent):

* **variable** — like for like only (temperature↔temperature, salinity↔salinity).
* **time**     — nearest model timestep; rejected if farther than a tolerance.
* **location** — nearest model grid cell; rejected if the float sits outside the
  model domain, or the nearest cell carries no data (land / analysis gap).
* **depth**    — each observed level is matched to the nearest model depth.

``difference = model_value - observed_value`` (positive ⇒ model reads higher than
the float). Nothing is fabricated: a level with no model value is dropped, and a
float that cannot be matched in time or space is reported in ``unmatched`` with a
reason rather than silently guessed.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

# Slice cache: (source_name, variable, model_time, depth) -> FieldSlice. A single
# comparison re-reads the same handful of depth slices across dozens of floats, so
# caching turns an O(floats x depths) slice storm into O(depths). Bounded to keep
# memory flat across many requests.
_SLICE_CACHE: dict[tuple, object] = {}
_SLICE_CACHE_MAX = 256


def _parse_time(value: str | None) -> datetime | None:
    """ISO-8601 -> naive-UTC datetime (obs times lack a 'Z', model times have one;
    normalising to tz-naive lets the two be compared directly)."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.replace(tzinfo=None)


def _nearest_index(values: list[float], target: float) -> int:
    arr = np.asarray(values, dtype=float)
    return int(np.argmin(np.abs(arr - float(target))))


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(2 * r * math.asin(math.sqrt(a)), 1)


def _cached_slice(model, variable: str, model_time: str, depth: float):
    key = (getattr(model, "name", "?"), variable, model_time, round(float(depth), 3))
    cached = _SLICE_CACHE.get(key)
    if cached is None:
        cached = model.slice(variable, depth=depth, time=model_time)
        if len(_SLICE_CACHE) >= _SLICE_CACHE_MAX:
            _SLICE_CACHE.clear()
        _SLICE_CACHE[key] = cached
    return cached


class _ModelColumn:
    """Nearest-grid model column at one location and timestep, sliced lazily.

    Holds the model's lat/lon axes (read from one probe slice) and resolves the
    nearest grid indices for a float once; each depth is then a cached slice read.
    """

    def __init__(self, model, variable: str, model_time: str, lat: float, lon: float):
        self._model = model
        self._variable = variable
        self._time = model_time
        probe = _cached_slice(model, variable, model_time, model.depths()[0])
        self._lat_axis = probe.lat
        self._lon_axis = probe.lon
        self._i = _nearest_index(probe.lat, lat)
        self._j = _nearest_index(probe.lon, lon)

    @property
    def grid_lat(self) -> float:
        return round(float(self._lat_axis[self._i]), 3)

    @property
    def grid_lon(self) -> float:
        return round(float(self._lon_axis[self._j]), 3)

    def in_domain(self, lat: float, lon: float) -> bool:
        """True if the float lies within ~one cell of the model's lat/lon extent."""
        la = np.asarray(self._lat_axis, dtype=float)
        lo = np.asarray(self._lon_axis, dtype=float)
        dlat = float(np.median(np.abs(np.diff(la)))) if la.size > 1 else 1.0
        dlon = float(np.median(np.abs(np.diff(lo)))) if lo.size > 1 else 1.0
        return (
            la.min() - dlat <= lat <= la.max() + dlat
            and lo.min() - dlon <= lon <= lo.max() + dlon
        )

    def value_at_depth(self, depth: float) -> tuple[float, float] | None:
        """(model_depth, model_value) at the nearest model depth, or None if no data."""
        depths = self._model.depths()
        di = _nearest_index(depths, depth)
        model_depth = float(depths[di])
        sl = _cached_slice(self._model, self._variable, self._time, model_depth)
        val = sl.values[self._i][self._j]
        if val is None:
            return None
        return model_depth, float(val)


def _nearest_model_time(model, obs_time: str) -> tuple[str | None, float | None]:
    """Nearest model timestep to an observation, plus the gap in days."""
    times = model.times()
    if not times:
        return None, None
    obs_dt = _parse_time(obs_time)
    if obs_dt is None:
        return times[0], None
    parsed = [(_parse_time(t), t) for t in times]
    parsed = [(dt, t) for dt, t in parsed if dt is not None]
    if not parsed:
        return times[0], None
    best_dt, best = min(parsed, key=lambda p: abs(p[0] - obs_dt))
    gap_days = round(abs((best_dt - obs_dt).total_seconds()) / 86400.0, 2)
    return best, gap_days


def _stats(diffs: list[float]) -> dict:
    a = np.asarray(diffs, dtype=float)
    return {
        "n": int(a.size),
        "mean_difference": round(float(a.mean()), 3),
        "rms_difference": round(float(np.sqrt((a**2).mean())), 3),
        "max_abs_difference": round(float(np.abs(a).max()), 3),
    }


def compare_float(model, obs_profile: dict, time_tolerance_days: float) -> dict:
    """Compare one Argo float profile against the model column above it.

    ``obs_profile`` is the dict from :meth:`ArgoInstrumentSource.profile`
    (``lat, lon, time, depths[], values[]`` for one variable). Returns either a
    matched comparison (paired levels + per-float stats) or ``{"matched": False,
    "reason": ...}``.
    """
    variable = obs_profile["variable"]
    lat, lon = obs_profile["lat"], obs_profile["lon"]
    obs_time = obs_profile.get("time") or ""

    model_time, gap_days = _nearest_model_time(model, obs_time)
    if model_time is None:
        return {"matched": False, "reason": "model has no timesteps"}
    if gap_days is not None and gap_days > time_tolerance_days:
        return {
            "matched": False,
            "reason": f"nearest model timestep is {gap_days:.0f} d away "
            f"(> {time_tolerance_days:.0f} d tolerance)",
        }

    column = _ModelColumn(model, variable, model_time, lat, lon)
    if not column.in_domain(lat, lon):
        return {"matched": False, "reason": "float is outside the model domain"}

    levels = []
    diffs = []
    for obs_depth, observed in zip(obs_profile["depths"], obs_profile["values"]):
        hit = column.value_at_depth(obs_depth)
        if hit is None:
            continue  # no model data at this cell/depth — skip, don't fabricate
        model_depth, model_value = hit
        diff = round(model_value - float(observed), 3)
        levels.append({
            "obs_depth": round(float(obs_depth), 1),
            "model_depth": round(model_depth, 1),
            "observed": round(float(observed), 3),
            "model": round(model_value, 3),
            "difference": diff,
        })
        diffs.append(diff)

    if not levels:
        return {"matched": False, "reason": "no overlapping depths with model data"}

    return {
        "matched": True,
        "float_id": obs_profile["id"],
        "name": obs_profile["name"],
        "lat": lat,
        "lon": lon,
        "observation_time": obs_time,
        "model_time": model_time,
        "time_diff_days": gap_days,
        "grid_lat": column.grid_lat,
        "grid_lon": column.grid_lon,
        "distance_km": _haversine_km(lat, lon, column.grid_lat, column.grid_lon),
        "levels": levels,
        "summary": _stats(diffs),
    }


def compare_argo(
    model,
    instruments,
    variable: str = "temperature",
    time_tolerance_days: float = 31.0,
    bbox: tuple[float, float, float, float] | None = None,
    float_id: str | None = None,
) -> dict:
    """Compare Argo observations against the model over a region.

    Iterates the floats (optionally one ``float_id`` or a ``bbox``), compares each
    to the model column above it, and rolls the per-float results into an overall
    bias / RMS summary. ``difference = model - observed`` throughout.
    """
    if variable not in model.variables():
        raise KeyError(variable)
    units = model.units(variable)

    if float_id is not None:
        try:
            floats = [instruments.get(float_id)]
        except KeyError:
            raise LookupError(float_id)
    else:
        floats = instruments.list(bbox)

    comparisons: list[dict] = []
    unmatched: list[dict] = []
    all_diffs: list[float] = []

    for inst in floats:
        if variable not in inst.variables:
            unmatched.append({"float_id": inst.id, "reason": f"float has no {variable}"})
            continue
        obs_profile = instruments.profile(inst.id, variable)
        result = compare_float(model, obs_profile, time_tolerance_days)
        if not result.get("matched"):
            unmatched.append({"float_id": inst.id, "reason": result["reason"]})
            continue
        comparisons.append(result)
        all_diffs.extend(d["difference"] for d in result["levels"])

    summary = {
        "floats_total": len(floats),
        "floats_matched": len(comparisons),
        "floats_unmatched": len(unmatched),
        "levels_matched": len(all_diffs),
    }
    if all_diffs:
        summary.update(_stats(all_diffs))

    return {
        "variable": variable,
        "units": units,
        "difference_convention": "model - observed (positive = model higher)",
        "model": {
            "source": getattr(model, "name", "?"),
            "time_tolerance_days": time_tolerance_days,
        },
        "observation": {"source": getattr(instruments, "name", "?")},
        "summary": summary,
        "comparisons": comparisons,
        "unmatched": unmatched,
    }
