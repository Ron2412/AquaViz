"""AquaViz API — FastAPI application entrypoint.

Parses NetCDF ocean-model files with xarray, slices them server-side by
variable / depth / time, and serves compact JSON to the frontend. Raw NetCDF is
never sent to the browser.

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.config import settings
from app.routers import comparison, fields, instruments

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(
    title="AquaViz API",
    version="0.1.0",
    description=(
        "Server-side slicing of ocean model + instrument data "
        "for browser-based 3D visualization."
    ),
)

# The frontend is a separate origin (Vite dev server), so CORS is required.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fields.router)
app.include_router(instruments.router)
app.include_router(comparison.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness probe used by the frontend and by container health checks."""
    return {"status": "ok", "service": "aquaviz-api", "version": app.version}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the self-contained 3D viewer (Three.js via CDN, same-origin fetches).

    This is an interim UI so the pipeline is viewable without an npm/Vite build;
    the React app in ``frontend/`` is the real Phase-1 client for when a package
    install is possible.
    """
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/api", tags=["meta"])
def api_info() -> dict:
    """Machine-readable index of available data endpoints."""
    return {
        "service": "aquaviz-api",
        "version": app.version,
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/meta/coords",
            "/field/{variable}",
            "/temperature",
            "/instruments",
            "/instruments/{id}/profile",
            "/api/comparison/argo",
            "/api/comparison/argo/{float_id}",
        ],
    }
