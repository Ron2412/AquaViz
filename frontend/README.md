# AquaViz frontend

React 18 + Three.js (r168) single-page app. Fetches server-sliced ocean fields
from the backend and renders them as an interactive 3D mesh colored by value.

## Setup

```bash
npm install
npm run dev        # http://localhost:5173
```

The backend must be running (default `http://localhost:8000`). Point elsewhere with:

```bash
VITE_API_URL=http://host:port npm run dev
```

## Phase 1 (current)

- `GET /meta/coords` builds the depth slider.
- `GET /temperature?depth=` drives a colored, orbitable 3D mesh (drag = orbit,
  scroll = zoom). A colorbar reflects the slice's value range.

## Layout

| File | Role |
|------|------|
| `src/api.js`       | Axios client + typed endpoint helpers |
| `src/OceanScene.js`| Imperative Three.js scene / mesh builder (React owns its lifecycle) |
| `src/colormap.js`  | Shared cool→warm ramp (mesh + colorbar) |
| `src/App.jsx`      | State, data fetching, depth control |
| `src/Colorbar.jsx` | Legend |

## Next phases

Time scrubber, salinity/currents, opacity + vertical-exaggeration controls,
clickable instrument markers → Chart.js depth profiles, 2D/3D toggle.
