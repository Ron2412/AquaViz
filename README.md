# AquaViz

Browser-native, WebGL-powered **3D ocean data visualization**. AquaViz co-renders
INCOIS ocean-model fields (temperature, salinity, currents) with instrument
observations (Argo floats, Gliders, CTD) in a single interactive 3D scene — with
no client-side installation.

## Architecture

| Layer     | Tech                                          | Role |
|-----------|-----------------------------------------------|------|
| Frontend  | React 18, Three.js, Chart.js, proj4js, Axios  | 3D mesh rendering + instrument profile panels |
| Backend   | Python 3.11+, FastAPI, xarray, netCDF4, NumPy | Server-side NetCDF slicing → lightweight JSON |
| Database  | PostgreSQL + PostGIS *(later phase)*          | Geospatial instrument storage |
| Packaging | Docker                                        | Deploy on INCOIS infrastructure |

Raw NetCDF files never reach the browser — the API slices data by
variable / depth / time and serves compact JSON.

## Repository layout

    backend/      FastAPI service (data ingestion + slicing API)
      app/        Application code
      data/       NetCDF model files (gitignored; populated by fetch scripts)
      scripts/    Data-acquisition scripts (public NOAA/Argo sources)
    frontend/     React + Three.js app (scaffolded with Vite in Phase 1)
    docs/         Roadmap and design notes

## Quickstart — backend

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# → http://127.0.0.1:8000/health
```

The frontend quickstart is added once the Vite app is scaffolded (Phase 1).

## Status

Build proceeds one phase at a time — see [docs/ROADMAP.md](docs/ROADMAP.md).
Currently: **Phase 0 — foundation**.
