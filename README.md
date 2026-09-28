# StormTrace

StormTrace is a map-first meteorological intelligence prototype. It compares real ECMWF forecast precipitation with a seasonal historical baseline, maps grid-point rainfall anomalies, groups adjacent threshold-exceeding cells, and summarizes how those detected regions change across forecast days. It is a screening tool, not an official warning service.

## Architecture and live data

- `main.py` serves the dashboard and FastAPI endpoints. The spatial pipeline uses Open-Meteo's ECMWF endpoint for hourly precipitation, 10 m wind, and 2 m temperature, then aggregates daily values in UTC. The existing point endpoints continue using the Open-Meteo Forecast API.
- Historical daily precipitation (2011–2020) comes from Open-Meteo Archive API. New requests explicitly select `models=era5`. Older cached values were fetched without that selector and remain labeled model-unverified. The baseline is a seasonal ±7 calendar-day window, with mean, standard deviation, and sample count calculated for each grid point/day.
- `weather-tracker.html` renders GeoJSON cells and connected regions with Leaflet over OpenStreetMap tiles. A selected forecast day controls its map field, region shapes, timeline and selected grid-cell metrics. Site choices snap to the nearest valid grid cell; map clicks select cells directly.
- `live_check.py` is the existing standard-library endpoint smoke checker. `main - Copy.py` is a legacy mock and is not used by the active application.

## Spatial grid

Defaults cover latitude 6–36°N and longitude 66–99°E at 3° spacing: 11 rows × 12 columns = **132 configured grid points**. This is a coarse regional overview grid, not ECMWF's native 9 km resolution. Coordinates and spacing can be changed with `STORMTRACE_MIN_LAT`, `STORMTRACE_MAX_LAT`, `STORMTRACE_MIN_LON`, `STORMTRACE_MAX_LON`, and `STORMTRACE_GRID_SPACING` environment variables. Requests are split into batches of up to 48 points (configurable with `STORMTRACE_BATCH_SIZE`), with at most two concurrent requests and 45-second timeouts. ECMWF refreshes use a three-hour cache TTL, aligned to the provider's six-hour model updates. The disk snapshot (`.stormtrace_spatial_cache.json`, git-ignored) remains available as a clearly marked cached fallback; after its TTL, the app attempts a refresh and serves the last real snapshot if that fails. On refresh, historical seasonal summaries are reused for the same cell/date; new 2011–2020 Archive requests explicitly select ERA5 and are limited to coordinates/dates missing from that cache. Rate-limit responses are reported, daily-limit failures are not retried, and transient 429 responses receive at most one provider-directed retry. API/dashboard metadata distinguishes `LIVE` from `CACHED`, reports source retrieval time and forecast timestamp range, and keeps partial coverage visible. Repeated D1–D10 selections reuse the cached analysis. Legacy fixed-location routes fall back to the nearest valid grid point during transient point-endpoint failure; unavailable hourly/probability fields remain empty.

## Anomaly and region methods

For a grid point and forecast date, rainfall percentage anomaly is `((forecast daily precipitation − historical seasonal mean) / historical seasonal mean) × 100`. The existing 0.1 mm reference guard is used when the baseline is ≤0.05 mm. The normalized extreme anomaly score is `(forecast − seasonal mean) / seasonal standard deviation`; it is reported as unavailable when the standard deviation is below 0.1 mm or source values are missing.

The default mask requires both percentage anomaly ≥50% and forecast daily precipitation ≥10 mm. These configurable thresholds (`STORMTRACE_ANOMALY_THRESHOLD_PCT`, `STORMTRACE_MIN_EXTREME_RAIN_MM`) are prototype screening rules, not official warning thresholds. Four-neighbor connected components form regions; components smaller than `STORMTRACE_MIN_REGION_CELLS` (default 2) are omitted. Approximate area sums the grid-cell areas. The API provides cell-based MultiPolygon geometry, centroid, bounds, area, peak/mean precipitation, mean anomaly and normalized score.

Adjacent-day regions are matched greedily by cell-overlap Jaccard similarity plus a proximity preference, with candidate matches gated by shared cells or centroid distance within 1.6 grid spacings. The resulting records classify persistence, movement, expansion, contraction, appearance and disappearance from the detected regions. This is **extreme-anomaly region evolution**, not operational storm tracking; coarse-grid movement and area are approximate.

## API

- `GET /api/spatial-field?day=1` — valid grid-cell GeoJSON properties, selected-day metrics, connected regions and evolution.
- `GET /api/threat-regions?day=1` — selected-day detected regions and matching events.
- `GET /api/threat-evolution` — all forecast-day regions and data-derived event records.
- `GET /api/threat-footprint?day=1[&region_id=...]` — selected-day region geometries as GeoJSON FeatureCollection.
- Existing endpoints remain: `/api/health`, `/api/forecast`, `/api/anomaly/{id}`, `/api/threat-zones`, `/api/tracking/{id}`, `/api/physics/{id}`, `/api/downscale/{id}`, `/api/uncertainty/{id}`, and `/api/alerts`.

## Run locally

Python 3.10+ is recommended. From PowerShell in the project folder:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/weather-tracker.html`. Internet access is required for the weather APIs, Leaflet, and map tiles. API documentation: `http://127.0.0.1:8000/docs`. Run the existing smoke checker from another terminal with `.venv\Scripts\python.exe live_check.py`.

## Implemented and limitations

**Implemented:** existing real point forecast and explicitly ERA5-selected baseline requests; ECMWF spatial forecast ingestion; configurable coarse grid; per-cell daily anomalies and normalized scores; threshold masks; connected cell regions and GeoJSON footprints; consecutive-day proximity/overlap matching; selected-day map, site-to-grid snapping, and cell selection; in-memory caching, retries and partial-feed status.

**Unavailable or not claimed:** ECMWF precipitation probability (not supplied by this endpoint); ensemble uncertainty; official warning classification; 12 km-to-5 km downscaling; trained GNN or diffusion inference; physics-based validation; operational storm-system tracking. The existing point APIs remain for compatibility and are not used as the spatial dashboard's source. Grid values are sampled at a deliberately coarse 3° display spacing.
