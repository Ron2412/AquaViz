"""Combine several instrument sources behind one roster.

The viewer shows a single set of clickable markers, but real deployments come
from separate archives — Argo floats and gliders live in different Ifremer
collections and parse differently (:class:`~app.data.argo.ArgoInstrumentSource`
reads Argo GDAC ``*_prof.nc``; the glider source reads EGO trajectory files).
This wrapper fans the instrument surface the router depends on — ``list`` /
``get`` / ``profile`` — across any number of sub-sources so they render as one
fleet. It holds no data of its own and makes no assumptions about a source's
file format, so a new instrument type only needs its own source class.

Dispatch is by ownership: a profile request goes to whichever sub-source knows
the id, so a bad *variable* on a known instrument still surfaces that source's
error (a 404) rather than being swallowed as "unknown instrument".
"""

from __future__ import annotations

from typing import Protocol


class _InstrumentSource(Protocol):
    """The slice of the instrument surface this wrapper fans out."""

    name: str

    def list(self, bbox: tuple[float, float, float, float] | None = None) -> list:
        ...

    def get(self, instrument_id: str):
        ...

    def profile(self, instrument_id: str, variable: str, time: str | None = None) -> dict:
        ...


class CombinedInstrumentSource:
    """Serve one roster + profiles drawn from several instrument sources."""

    name = "combined"

    def __init__(self, sources: list[_InstrumentSource]) -> None:
        # Drop any that failed to construct (a missing/broken file loads as None).
        self._sources = [s for s in sources if s is not None]

    def _owner(self, instrument_id: str) -> _InstrumentSource | None:
        """First sub-source that recognises ``instrument_id``, else ``None``."""
        for src in self._sources:
            try:
                src.get(instrument_id)
                return src
            except KeyError:
                continue
        return None

    # -- catalogue -----------------------------------------------------------
    def list(self, bbox: tuple[float, float, float, float] | None = None) -> list:
        """Merged roster across every sub-source (first wins on id collision)."""
        out: list = []
        seen: set[str] = set()
        for src in self._sources:
            for inst in src.list(bbox):
                if inst.id in seen:
                    continue
                seen.add(inst.id)
                out.append(inst)
        return out

    def get(self, instrument_id: str):
        owner = self._owner(instrument_id)
        if owner is None:
            raise KeyError(instrument_id)
        return owner.get(instrument_id)

    # -- profiles ------------------------------------------------------------
    def profile(self, instrument_id: str, variable: str, time: str | None = None) -> dict:
        owner = self._owner(instrument_id)
        if owner is None:
            raise KeyError(instrument_id)
        # Owner found: any KeyError now is a bad variable, which must propagate.
        return owner.profile(instrument_id, variable, time)
