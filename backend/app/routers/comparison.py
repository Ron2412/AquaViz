"""Model-vs-observation comparison endpoints.

The platform's headline question, over REST: for Argo floats in a region, what did
the model estimate at each float's place/time/depth, and how far off was it?

``/api/comparison/argo`` compares every float in a bounding box; the
``/{float_id}`` variant compares a single float (for the click-a-marker profile
overlay). Both delegate to :func:`app.comparison.compare_argo`, which matches
observation→model on time, location, depth and variable and reports
``model - observed``.

The model defaults to the richest source whose timesteps *overlap* the loaded
observations (:func:`app.data.get_comparison_model`) — a matchup is only
meaningful when model and obs coexist in time. A ``model=hycom|glorys|incois``
query overrides that (e.g. to force the HYCOM forecast and see the honest
time-tolerance rejection against historical floats). This router never assumes
which model is active; each response labels the one it used.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.comparison import compare_argo
from app.data import (
    available_models,
    get_comparison_model,
    get_instruments,
    get_model,
)

router = APIRouter(tags=["comparison"])
_instruments = get_instruments()
# Default model: the one that overlaps the observations in time (else primary).
_default_model = get_comparison_model(_instruments)


def _parse_bbox(bbox: str | None) -> tuple[float, float, float, float] | None:
    if not bbox:
        return None
    try:
        parts = [float(x) for x in bbox.split(",")]
        if len(parts) != 4:
            raise ValueError
    except ValueError:
        raise HTTPException(
            status_code=400, detail="bbox must be 'min_lon,min_lat,max_lon,max_lat'"
        )
    return (parts[0], parts[1], parts[2], parts[3])


def _resolve_model(model: str | None):
    """The requested model source, or the time-overlap default. 404 if unknown."""
    if not model or model == "auto":
        return _default_model
    try:
        return get_model(model)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown or unavailable model {model!r}. "
            f"Available: {['auto', *available_models()]}",
        )


def _run(
    variable: str, tolerance: float, bbox: str | None, float_id: str | None, model: str | None
) -> dict:
    source = _resolve_model(model)
    try:
        return compare_argo(
            source,
            _instruments,
            variable=variable,
            time_tolerance_days=tolerance,
            bbox=_parse_bbox(bbox),
            float_id=float_id,
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Variable {variable!r} not available in model {source.name!r}",
        )
    except LookupError:
        raise HTTPException(status_code=404, detail=f"No such float: {float_id!r}")


@router.get("/api/comparison/argo")
def comparison_argo(
    variable: str = Query("temperature", description="temperature or salinity."),
    time_tolerance_days: float = Query(
        31.0, gt=0, description="Reject a float if the nearest model timestep is farther."
    ),
    bbox: str | None = Query(
        None, description="Restrict floats to 'min_lon,min_lat,max_lon,max_lat'."
    ),
    model: str | None = Query(
        None, description="Model to compare against: auto (default), hycom, glorys, incois."
    ),
) -> dict:
    """Compare all Argo floats in a region against the model (``model - observed``)."""
    return _run(variable, time_tolerance_days, bbox, None, model)


@router.get("/api/comparison/argo/{float_id}")
def comparison_argo_float(
    float_id: str,
    variable: str = Query("temperature", description="temperature or salinity."),
    time_tolerance_days: float = Query(31.0, gt=0),
    model: str | None = Query(
        None, description="Model to compare against: auto (default), hycom, glorys, incois."
    ),
) -> dict:
    """Compare one Argo float against the model column above it (profile overlay)."""
    return _run(variable, time_tolerance_days, None, float_id, model)
