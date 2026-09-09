"""Field-slicing endpoints: server-side depth/time slices as lightweight JSON.

Raw NetCDF never reaches the browser — the client receives only the 2-D grid it
needs to render one mesh. ``/field/{variable}`` is the general slice endpoint
(temperature, salinity, current_speed); ``/temperature`` remains as a Phase-1
alias. Instrument endpoints follow in Phase 3.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.data import get_source
from app.data.base import column_via_slices

router = APIRouter(tags=["fields"])
_source = get_source()


@router.get("/meta/coords", tags=["meta"])
def coords() -> dict:
    """Coordinate catalogue for the frontend to build sliders and scrubbers.

    ``variables`` carries per-variable units and absolute colour domains so the
    client can label the legend and colour every slice on a fixed scale.
    """
    variables = {
        name: {"units": _source.units(name), "domain": _source.domain(name)}
        for name in _source.variables()
    }
    return {
        "source": _source.name,
        "variables": variables,
        "depths": _source.depths(),
        "times": _source.times(),
    }


def _slice(variable: str, depth: float, time: str | None) -> dict:
    try:
        result = _source.slice(variable, depth=depth, time=time)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Variable {variable!r} not available"
        )
    return result.to_dict()


@router.get("/field/{variable}")
def field(
    variable: str,
    depth: float = Query(0.0, ge=0, description="Depth in metres (snaps to nearest level)."),
    time: str | None = Query(None, description="ISO-8601 timestep (defaults to first)."),
) -> dict:
    """One variable sliced to a depth and time as a lat×lon grid."""
    return _slice(variable, depth, time)


@router.get("/volume/{variable}")
def volume(
    variable: str,
    time: str | None = Query(None, description="ISO-8601 timestep (defaults to first)."),
    max_levels: int = Query(
        12, ge=2, le=24, description="Max depth levels to stack (subsampled evenly)."
    ),
) -> dict:
    """A depth-stacked, bbox-trimmed volume of ``variable`` for one timestep.

    Powers the frontend's concentric depth shells: one lat×lon grid per sampled
    depth level, cropped to the data's bounding box so the water column ships as
    a compact stack instead of many full-globe slices. Uses the source's native
    ``column()`` when available, else the generic slice-and-trim fallback.
    """
    try:
        if hasattr(_source, "column"):
            return _source.column(variable, time=time, max_levels=max_levels)
        return column_via_slices(_source, variable, time=time, max_levels=max_levels)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Variable {variable!r} not available"
        )


@router.get("/temperature")
def temperature(
    depth: float = Query(0.0, ge=0, description="Depth in metres (snaps to nearest level)."),
    time: str | None = Query(None, description="ISO-8601 timestep (defaults to first)."),
) -> dict:
    """Temperature slice (Phase-1 alias for ``/field/temperature``)."""
    return _slice("temperature", depth, time)
