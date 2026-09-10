"""Tests for the model-vs-observation comparison logic (:mod:`app.comparison`).

These lock the *scientific* contract the platform rests on:

* ``difference = model - observed`` (positive means the model reads higher) — in
  both directions,
* each observed level is matched to the **nearest** model depth,
* a float too far from any model timestep, or outside the model's domain, is
  reported as ``unmatched`` **with a reason** rather than silently guessed,
* levels with no model value are dropped, never fabricated,
* the roll-up bias / RMS / max-|Δ| statistics are computed correctly.

They use tiny in-memory fakes for the model and the instruments, so they need no
NetCDF files and run in milliseconds. pytest isn't installed in the offline dev
venv, so this file is also runnable directly::

    backend/.venv/bin/python backend/tests/test_comparison.py

which executes every ``test_*`` function and prints a PASS/FAIL line each. When
pytest *is* available, ``pytest backend/tests`` collects the same functions.
"""

from __future__ import annotations

import os
import sys

# Make ``app`` importable whether run via pytest or directly as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.comparison import _SLICE_CACHE, compare_argo, compare_float  # noqa: E402
from app.data.base import FieldSlice  # noqa: E402

_UNITS = {"temperature": "degree_C", "salinity": "PSU"}


class FakeModel:
    """Minimal :class:`OceanDataSource`: a uniform lat×lon grid per depth.

    ``grid`` maps a depth (m) to a 2-D list of values shaped ``lat × lon`` (use
    ``None`` for a land / no-data cell). ``slice`` snaps the requested depth to
    the nearest level, exactly as a real reader does.
    """

    def __init__(self, times, depths, lat, lon, grid):
        self.name = "fake-model"
        self._times = list(times)
        self._depths = list(depths)
        self._lat = list(lat)
        self._lon = list(lon)
        self._grid = grid

    def variables(self):
        return ["temperature", "salinity"]

    def units(self, v):
        return _UNITS[v]  # KeyError for an unknown variable, like the real readers

    def depths(self):
        return list(self._depths)

    def times(self):
        return list(self._times)

    def slice(self, variable, depth, time=None):
        if variable not in _UNITS:
            raise KeyError(variable)
        di = min(range(len(self._depths)), key=lambda i: abs(self._depths[i] - depth))
        d = self._depths[di]
        return FieldSlice(
            variable=variable,
            units=self.units(variable),
            depth=float(d),
            depth_requested=float(depth),
            time=time or self._times[0],
            time_index=0,
            lat=list(self._lat),
            lon=list(self._lon),
            values=self._grid[d],
            value_min=None,
            value_max=None,
        )


class FakeInstrument:
    def __init__(self, id, lat, lon, time, variables):
        self.id = id
        self.kind = "argo"
        self.name = f"Argo {id}"
        self.lat = lat
        self.lon = lon
        self.time = time
        self.variables = tuple(variables)


class FakeInstruments:
    """Minimal instrument source over pre-built profiles."""

    name = "fake-obs"

    def __init__(self, insts, profiles):
        self._insts = list(insts)
        self._by_id = {i.id: i for i in insts}
        self._profiles = profiles  # {id: {variable: (depths[], values[])}}

    def list(self, bbox=None):
        if bbox is None:
            return list(self._insts)
        mnlon, mnlat, mxlon, mxlat = bbox
        return [
            i for i in self._insts
            if mnlon <= i.lon <= mxlon and mnlat <= i.lat <= mxlat
        ]

    def get(self, id):
        return self._by_id[id]  # KeyError for an unknown id -> router 404

    def profile(self, id, variable, time=None):
        inst = self._by_id[id]
        depths, values = self._profiles[id][variable]
        return {
            "id": inst.id,
            "name": inst.name,
            "kind": inst.kind,
            "lat": inst.lat,
            "lon": inst.lon,
            "variable": variable,
            "units": _UNITS[variable],
            "time": inst.time,
            "depths": list(depths),
            "values": list(values),
        }


def _uniform(lat, lon, value):
    """A lat×lon grid with every cell set to ``value`` (or ``None``)."""
    return [[value for _ in lon] for _ in lat]


def _model(grid_by_depth):
    """A 3×3-cell model at one timestep with the given depth→value map."""
    lat = [0.0, 1.0, 2.0]
    lon = [10.0, 11.0, 12.0]
    depths = sorted(grid_by_depth)
    grid = {d: _uniform(lat, lon, grid_by_depth[d]) for d in depths}
    return FakeModel(["2022-01-15T00:00:00Z"], depths, lat, lon, grid)


def _reset():
    """Clear the module-global slice cache so tests can reuse model names."""
    _SLICE_CACHE.clear()


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #

def test_difference_is_model_minus_observed_positive():
    """Model warmer than the float -> positive difference; depths snap to nearest."""
    _reset()
    model = _model({0.0: 20.0, 10.0: 18.0, 100.0: 5.0})
    obs = {
        "id": "f1", "name": "Argo f1", "variable": "temperature",
        "lat": 1.0, "lon": 11.0, "time": "2022-01-10T00:00:00",
        "depths": [0.0, 12.0], "values": [19.0, 17.0],
    }
    res = compare_float(model, obs, time_tolerance_days=31.0)
    assert res["matched"] is True, res
    # 12 m observed level snaps to the 10 m model level.
    assert [lv["model_depth"] for lv in res["levels"]] == [0.0, 10.0]
    # difference = model - observed = 20-19 and 18-17 = +1.0 each.
    assert [lv["difference"] for lv in res["levels"]] == [1.0, 1.0]
    assert res["summary"]["mean_difference"] == 1.0
    assert res["summary"]["max_abs_difference"] == 1.0


def test_difference_negative_when_model_cooler():
    """Model colder than the float -> negative difference (sign locked both ways)."""
    _reset()
    model = _model({0.0: 15.0})
    obs = {
        "id": "f1", "name": "Argo f1", "variable": "temperature",
        "lat": 1.0, "lon": 11.0, "time": "2022-01-15T00:00:00",
        "depths": [0.0], "values": [18.0],
    }
    res = compare_float(model, obs, time_tolerance_days=31.0)
    assert res["matched"] is True
    assert res["levels"][0]["difference"] == -3.0  # 15 - 18


def test_time_tolerance_rejects_far_float():
    """A float far from every model timestep is unmatched, with a reason."""
    _reset()
    model = _model({0.0: 20.0})
    obs = {
        "id": "f1", "name": "Argo f1", "variable": "temperature",
        "lat": 1.0, "lon": 11.0, "time": "2020-01-01T00:00:00",  # ~2 yr away
        "depths": [0.0], "values": [19.0],
    }
    res = compare_float(model, obs, time_tolerance_days=31.0)
    assert res["matched"] is False
    assert "tolerance" in res["reason"]


def test_out_of_domain_rejected():
    """A float outside the model's lat/lon extent is unmatched, with a reason."""
    _reset()
    model = _model({0.0: 20.0})
    obs = {
        "id": "f1", "name": "Argo f1", "variable": "temperature",
        "lat": 50.0, "lon": 200.0, "time": "2022-01-15T00:00:00",
        "depths": [0.0], "values": [19.0],
    }
    res = compare_float(model, obs, time_tolerance_days=31.0)
    assert res["matched"] is False
    assert "domain" in res["reason"]


def test_no_model_data_levels_dropped_not_fabricated():
    """A no-data (None) model cell yields no level, and no fabricated value."""
    _reset()
    model = _model({0.0: None})  # nearest cell carries no data
    obs = {
        "id": "f1", "name": "Argo f1", "variable": "temperature",
        "lat": 1.0, "lon": 11.0, "time": "2022-01-15T00:00:00",
        "depths": [0.0], "values": [19.0],
    }
    res = compare_float(model, obs, time_tolerance_days=31.0)
    assert res["matched"] is False
    assert "no overlapping depths" in res["reason"]


def test_compare_argo_rolls_up_stats_and_convention():
    """The region roll-up reports the right counts, stats, units and convention."""
    _reset()
    model = _model({0.0: 20.0})
    insts = [
        FakeInstrument("f1", 1.0, 11.0, "2022-01-15T00:00:00", ["temperature"]),
        FakeInstrument("f2", 0.0, 10.0, "2022-01-15T00:00:00", ["temperature"]),
        FakeInstrument("far", 1.0, 11.0, "2010-01-01T00:00:00", ["temperature"]),
    ]
    profiles = {
        "f1": {"temperature": ([0.0], [19.0])},   # diff +1.0
        "f2": {"temperature": ([0.0], [22.0])},   # diff -2.0
        "far": {"temperature": ([0.0], [19.0])},  # rejected on time
    }
    obs = FakeInstruments(insts, profiles)
    out = compare_argo(model, obs, variable="temperature", time_tolerance_days=31.0)

    assert out["difference_convention"] == "model - observed (positive = model higher)"
    assert out["units"] == "degree_C"
    assert out["model"]["source"] == "fake-model"
    assert out["observation"]["source"] == "fake-obs"
    s = out["summary"]
    assert s["floats_total"] == 3
    assert s["floats_matched"] == 2
    assert s["floats_unmatched"] == 1
    assert s["levels_matched"] == 2
    # diffs [+1.0, -2.0]: mean -0.5, rms sqrt((1+4)/2)=1.581, max|Δ| 2.0
    assert s["mean_difference"] == -0.5
    assert s["max_abs_difference"] == 2.0
    assert abs(s["rms_difference"] - 1.581) < 1e-3


def test_unknown_variable_raises_keyerror():
    """A variable the model doesn't carry raises KeyError (router -> 404)."""
    _reset()
    model = _model({0.0: 20.0})
    obs = FakeInstruments([], {})
    try:
        compare_argo(model, obs, variable="chlorophyll")
    except KeyError:
        return
    raise AssertionError("expected KeyError for an unknown variable")


def test_float_missing_variable_is_unmatched():
    """A float without the requested variable is reported, not compared."""
    _reset()
    model = _model({0.0: 35.0})
    insts = [FakeInstrument("f1", 1.0, 11.0, "2022-01-15T00:00:00", ["temperature"])]
    obs = FakeInstruments(insts, {"f1": {"temperature": ([0.0], [30.0])}})
    out = compare_argo(model, obs, variable="salinity", time_tolerance_days=31.0)
    assert out["summary"]["floats_matched"] == 0
    assert out["unmatched"][0]["float_id"] == "f1"
    assert "salinity" in out["unmatched"][0]["reason"]


def test_unknown_float_id_raises_lookuperror():
    """Requesting a nonexistent float id raises LookupError (router -> 404)."""
    _reset()
    model = _model({0.0: 20.0})
    obs = FakeInstruments([], {})
    try:
        compare_argo(model, obs, variable="temperature", float_id="ghost")
    except LookupError:
        return
    raise AssertionError("expected LookupError for an unknown float id")


# --------------------------------------------------------------------------- #
# Plain-python runner (used when pytest isn't installed)                       #
# --------------------------------------------------------------------------- #

def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for t in tests:
        try:
            t()
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            failures += 1
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS  {t.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
