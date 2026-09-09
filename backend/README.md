# AquaViz backend

FastAPI service that parses NetCDF ocean-model files with xarray, slices them
server-side by variable / depth / time, and serves lightweight JSON to the
frontend. Raw NetCDF is never sent to the browser.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

- `GET /health` — liveness probe
- `GET /docs`   — interactive OpenAPI docs

## Endpoints (planned)

| Endpoint                     | Returns |
|------------------------------|---------|
| `/temperature?depth=&time=`  | sliced temperature grid (JSON) |
| `/salinity?depth=&time=`     | sliced salinity grid (JSON) |
| `/currents?depth=&time=`     | current vector grid (JSON) |
| `/floats?bbox=`              | Argo / Glider positions within a bounding box |
| `/float/{id}/profile`        | full depth profile for one instrument |

## Data

NetCDF files live in `backend/data/` (gitignored). Scripts under `scripts/`
populate it from public sources (NOAA ERDDAP / NCEI, Argo GDAC). A modular
loader isolates the source so real INCOIS files can be dropped in unchanged.
