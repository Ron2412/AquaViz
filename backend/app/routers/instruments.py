"""Instrument endpoints: platform locations + depth profiles.

Phase-3 half of the PRD — co-rendering in-situ observations (Argo floats,
Gliders, CTD) with the model fields. ``/instruments`` places clickable markers
on the globe; ``/instruments/{id}/profile`` feeds the depth-vs-value chart.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.data import get_instruments

router = APIRouter(tags=["instruments"])
_instruments = get_instruments()


@router.get("/instruments")
def list_instruments(
    bbox: str | None = Query(
        None,
        description="Filter to 'min_lon,min_lat,max_lon,max_lat' (WGS84 degrees).",
    ),
) -> dict:
    """Instrument roster (optionally filtered to a bounding box)."""
    parsed: tuple[float, float, float, float] | None = None
    if bbox:
        try:
            parts = [float(x) for x in bbox.split(",")]
            if len(parts) != 4:
                raise ValueError
            parsed = (parts[0], parts[1], parts[2], parts[3])
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="bbox must be 'min_lon,min_lat,max_lon,max_lat'",
            )
    items = _instruments.list(parsed)
    return {"count": len(items), "instruments": [i.to_dict() for i in items]}


@router.get("/instruments/{instrument_id}/profile")
def instrument_profile(
    instrument_id: str,
    variable: str = Query("temperature", description="Variable to profile."),
    time: str | None = Query(None, description="ISO-8601 timestep (defaults to first)."),
) -> dict:
    """Depth profile of ``variable`` at one instrument (depth-vs-value)."""
    try:
        return _instruments.profile(instrument_id, variable, time)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"No profile for instrument/variable: {exc.args[0]!r}",
        )
