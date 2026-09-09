"""Runtime configuration for the AquaViz API.

Values come from environment variables with sensible local-dev defaults, so the
same container image can be reconfigured per deployment without code changes.
"""

import os

_BACKEND_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class Settings:
    # Directory holding NetCDF model files (populated by scripts/, gitignored).
    data_dir: str = os.path.abspath(
        os.getenv("AQUAVIZ_DATA_DIR", os.path.join(_BACKEND_ROOT, "data"))
    )

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
