"""Runtime configuration for the AquaViz API.

Values come from environment variables with sensible local-dev defaults, so the
same container image can be reconfigured per deployment without code changes.
"""

import os

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _abs_or_none(value: str | None) -> str | None:
    """Absolute path for a configured file, or None when unset/blank."""
    value = (value or "").strip()
    return os.path.abspath(os.path.expanduser(value)) if value else None


class Settings:
    # Directory holding NetCDF model files (populated by scripts/, gitignored).
    data_dir: str = os.path.abspath(
        os.getenv("AQUAVIZ_DATA_DIR", os.path.join(_BACKEND_ROOT, "data"))
    )

    # Large ocean-MODEL file (e.g. the ~10 GB HYCOM NetCDF). Kept OUTSIDE the
    # repo and never committed; point MODEL_DATA_PATH at wherever it lives. When
    # unset, the model source auto-discovers ``*hycom*.nc`` in ``data_dir``.
    model_data_path: str | None = _abs_or_none(os.getenv("MODEL_DATA_PATH"))

    # Argo OBSERVATION file/dir (the ``*_prof.nc`` floats). When unset, the Argo
    # source auto-discovers ``*_prof.nc`` in ``data_dir`` (existing behaviour).
    argo_data_path: str | None = _abs_or_none(os.getenv("ARGO_DATA_PATH"))

    # Frontend origins allowed to call the API (Vite dev server defaults to 5173).
    cors_origins: list[str] = [
        origin.strip()
        for origin in os.getenv(
            "AQUAVIZ_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ]


settings = Settings()
