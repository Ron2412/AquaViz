"""Synthetic global field nudged toward real Argo observations.

The synthetic source gives a complete, pretty global field but is fictional; the
Argo file gives real measurements but only where floats happen to be (here, the
Indian/Southern Ocean). This source keeps the synthetic field everywhere and
*corrects* it toward the real Argo values near each float, fading smoothly back
to synthetic with distance — so there are no grey holes, and the observed region
reflects measured temperature/salinity.

Method (a lightweight Cressman / Gaussian nudge):

    blended = synthetic + Σ_i w_i · (obs_i − synthetic_at_float_i)
                          ─────────────────────────────────────────
                                        Σ_i w_i

where ``w_i = exp(−d_i² / 2σ²)`` and ``d_i`` is the great-circle distance from a
grid cell to float *i*. The correction is the observation-minus-synthetic
*innovation* at each float, distance-weighted — so far from every float the
weights vanish and the result is the untouched synthetic field.

Only variables Argo measures (temperature, salinity) are blended. Everything
else (e.g. current_speed) passes straight through to the synthetic source, and
so do :meth:`variables`, :meth:`units`, :meth:`depths`, and :meth:`times`.
"""

from __future__ import annotations

import numpy as np

from .argo import ArgoInstrumentSource
from .base import FieldSlice
from .synthetic import SyntheticOceanSource

# Variables Argo actually observes; others are passed through unblended.
_BLENDABLE = ("temperature", "salinity")

# Correlation length of the nudge (kilometres). Roughly the radius over which a
# float's correction is felt; ~800 km is a common ocean mesoscale-to-basin
# influence scale and keeps the corrected patch coherent without smearing
# globally.
_SIGMA_KM = 800.0

# Below this weight a float contributes nothing to a cell (keeps far cells exactly
# synthetic and lets us skip negligible work).
_MIN_WEIGHT = 1e-3

_EARTH_R_KM = 6371.0


class BlendedOceanSource:
    """Synthetic field corrected toward Argo observations near each float."""

    name = "synthetic + real Argo"

    def __init__(self, synthetic: SyntheticOceanSource, argo: ArgoInstrumentSource) -> None:
        self._syn = synthetic
        self._argo = argo
        self._domains: dict[str, dict[str, float]] = {}
        # Per-variable float geometry + profiles, prepared once.
        # variable -> (lat[N], lon[N], list_of(depths[], values[]))
        self._floats: dict[str, tuple[np.ndarray, np.ndarray, list]] = {}
        for var in _BLENDABLE:
            profs = argo.profiles_for(var)
            if not profs:
                continue
            lats = np.array([p[0] for p in profs], dtype=float)
            lons = np.array([p[1] for p in profs], dtype=float)
            series = [(p[2], p[3]) for p in profs]  # (depths, values) per float
            self._floats[var] = (lats, lons, series)

        # Warm the colour domains now so the first /meta/coords request is fast
        # (blending every depth/time is a few seconds; pay it once at startup).
        for var in _BLENDABLE:
            if var in self._floats:
                self.domain(var)

    # -- catalogue (delegate to synthetic) -----------------------------------
    def variables(self) -> list[str]:
        return self._syn.variables()

    def units(self, variable: str) -> str:
        return self._syn.units(variable)

    def depths(self) -> list[float]:
        return self._syn.depths()

    def times(self) -> list[str]:
        return self._syn.times()

    def domain(self, variable: str) -> dict[str, float]:
        """Value range of the *blended* field across all depths/times.

        Blending can pull values slightly beyond the synthetic range where
        observations are more extreme, so recompute rather than trusting the
        synthetic domain. Cached per variable. Non-blended variables defer to
        the synthetic domain directly.
        """
        if variable not in self._floats:
            return self._syn.domain(variable)
        if variable not in self._domains:
            lo, hi = float("inf"), float("-inf")
            for t_index in range(len(self._syn.times())):
                time = self._syn.times()[t_index]
                for depth in self._syn.depths():
                    grid = self._blended_grid(variable, float(depth), time)
                    lo = min(lo, float(np.nanmin(grid)))
                    hi = max(hi, float(np.nanmax(grid)))
            self._domains[variable] = {"min": round(lo, 3), "max": round(hi, 3)}
        return dict(self._domains[variable])

    # -- blending ------------------------------------------------------------
    def _obs_at_depth(self, series: list, depth: float) -> np.ndarray:
        """Each float's measured value at ``depth`` (NaN if the float can't say).

        Within a float's sampled span we linearly interpolate. Above its
        shallowest sample (e.g. the 0 m surface vs a first reading at ~2 m) we
        clamp to that shallowest value — the surface sits in the mixed layer, so
        this is physically safe and lets the most-viewed depth (0 m) show real
        data. Below its deepest sample we return NaN rather than inventing a
        value, so a shallow float doesn't constrain the abyss.
        """
        out = np.empty(len(series), dtype=float)
        for i, (depths, values) in enumerate(series):
            if depths.size == 0 or depth > depths[-1]:
                out[i] = np.nan
            elif depth <= depths[0]:
                out[i] = float(values[0])          # clamp to surface reading
            else:
                out[i] = float(np.interp(depth, depths, values))
        return out

    def _blended_grid(self, variable: str, depth: float, time: str | None) -> np.ndarray:
        """Synthetic grid at (depth, time) blended toward Argo obs.

        Convex combination ``(1−α)·synth + α·obs`` where ``α`` is the closest
        float's Gaussian weight (→1 at a float, →0 far away) and ``obs`` is the
        distance-weighted mean of the floats' readings. Being a convex blend, the
        result is always bounded by the synthetic and observed values — it can't
        overshoot into unphysical territory the way an additive nudge can.
        """
        actual_depth = self._syn._nearest_depth(depth)
        t_index = self._syn._resolve_time(time)
        field_fn = self._syn._fields[variable][1]
        base = field_fn(actual_depth, t_index)  # (nlat, nlon)

        lats, lons, series = self._floats[variable]
        obs = self._obs_at_depth(series, actual_depth)     # (N,)
        valid = np.isfinite(obs)
        if not valid.any():
            return base  # no float reaches this depth → pure synthetic

        f_lat = lats[valid]
        f_lon = lons[valid]
        obs = obs[valid]

        # Accumulate distance-weighted obs and track the strongest single pull
        # (α) so the blend is a proper convex combination.
        lat2d = self._syn._lat2d                            # (nlat, nlon)
        lon2d = self._syn._lon2d
        weight_sum = np.zeros_like(base)
        obs_sum = np.zeros_like(base)
        alpha = np.zeros_like(base)                         # max single weight
        inv_2sigma2 = 1.0 / (2.0 * _SIGMA_KM ** 2)
        for k in range(f_lat.size):
            d_km = _haversine_km(lat2d, lon2d, f_lat[k], f_lon[k])
            w = np.exp(-(d_km ** 2) * inv_2sigma2)
            obs_sum += w * obs[k]
            weight_sum += w
            np.maximum(alpha, w, out=alpha)

        out = base.copy()
        mask = weight_sum > _MIN_WEIGHT
        obs_field = np.where(mask, obs_sum / np.where(mask, weight_sum, 1.0), base)
        out[mask] = (1.0 - alpha[mask]) * base[mask] + alpha[mask] * obs_field[mask]
        return out

    # -- OceanDataSource -----------------------------------------------------
    def slice(self, variable: str, depth: float, time: str | None) -> FieldSlice:
        if variable not in self._floats:
            # Not observed by Argo → plain synthetic slice.
            return self._syn.slice(variable, depth, time)

        try:
            units = self._syn.units(variable)
        except KeyError:
            raise KeyError(variable)

        actual_depth = self._syn._nearest_depth(depth)
        t_index = self._syn._resolve_time(time)
        grid = self._blended_grid(variable, actual_depth, self._syn.times()[t_index])

        values = [[FieldSlice.clean(v) for v in row] for row in grid]
        return FieldSlice(
            variable=variable,
            units=units,
            depth=actual_depth,
            depth_requested=float(depth),
            time=self._syn.times()[t_index],
            time_index=t_index,
            lat=[float(v) for v in self._syn.lat],
            lon=[float(v) for v in self._syn.lon],
            values=values,
            value_min=round(float(np.nanmin(grid)), 3),
            value_max=round(float(np.nanmax(grid)), 3),
        )


def _haversine_km(lat1, lon1, lat2: float, lon2: float):
    """Great-circle distance (km) from grid arrays ``(lat1,lon1)`` to a point."""
    rlat1 = np.radians(lat1)
    rlat2 = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * _EARTH_R_KM * np.arcsin(np.sqrt(a))
