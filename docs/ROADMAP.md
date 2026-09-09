# AquaViz roadmap

An incremental build. Each phase is independently runnable and reviewable, so we
always have something working rather than a half-wired big bang.

## Success criteria (from the PRD)

- A depth slice loads in the 3D view in **< 3 s** on a standard workstation.
- Instrument marker click → profile chart in **< 1 s**.
- Variables: temperature, salinity, current vectors (minimum).
- Instruments: Argo floats, Gliders, CTD.
- Zero client-side install (browser only); deployable via Docker on INCOIS infra.

## Phases

### Phase 0 — Foundation & environment  ✅ done
- Repo structure, `.gitignore`, docs.
- Backend dependencies (FastAPI, xarray, netCDF4, NumPy, Uvicorn) + a `/health` endpoint.
- Sandbox network allowlist for package registries + public data hosts.

### Phase 1 — Vertical slice: temperature → 3D globe  ✅ done
- Synthetic CF-shaped temperature field (depth × lat × lon × time) via a modular
  `OceanDataSource` (NetCDF source drops in later with no router/frontend change).
- `GET /temperature?depth=&time=` — server-side slice → compact JSON grid.
- Three.js **3D globe** (data draped on a sphere, real coastlines, atmosphere,
  starfield, orbit/zoom/auto-spin) served from the API — no npm/Vite build needed.
- Depth slider + colorbar on a fixed absolute domain. **Proves the pipeline end-to-end.**

  _Interim note:_ frontend runs from `backend/app/static/index.html` (Three.js via
  CDN) because npm install is unavailable offline. The React app in `frontend/`
  is the eventual client and mirrors the same logic.

### Phase 2 — Variables & controls  ✅ done
- `salinity` and `current_speed` fields via a variable registry in the source.
- `GET /field/{variable}?depth=&time=` (temperature/salinity/current_speed);
  `/temperature` kept as an alias.
- `/meta/coords` now advertises per-variable units + absolute colour domains.
- Frontend variable selector + time scrubber; colorbar relabels/rescales per variable.
- _Deferred to a later polish pass:_ colorbar editor (palette, min/max, log/linear),
  current-vector arrows (currently rendered as speed magnitude).

### Phase 3 — Instruments  ← current
- Argo / Glider / CTD markers as clickable 3D objects in the scene.
- Click marker → `/float/{id}/profile` → Chart.js depth-vs-variable panel.
- `/floats?bbox=` for region filtering.

### Phase 4 — Persistence
- PostgreSQL + PostGIS for instrument positions and profile readings.
- Geospatial bbox queries; indexes on latitude / longitude / time.

### Phase 5 — Visualization polish
- Layer opacity, vertical exaggeration, 2D depth-slice / 3D view toggle.

### Phase 6 — Packaging & deployment
- Dockerfiles (backend, frontend), compose, INCOIS deployment notes.

## Data sources (public)

Development uses public, CF-compliant ocean data, fetched server-side:
- **Model fields:** NOAA ERDDAP / NCEI (e.g. World Ocean Atlas, HYCOM).
- **Instruments:** Argo GDAC profile data.

A modular loader isolates the data source, so real INCOIS files drop in without
touching the API or frontend.
