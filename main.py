"""StormTrace real-data forecast and spatial-anomaly prototype.

The application serves Open-Meteo ECMWF spatial forecasts, a seasonal historical
rainfall baseline, grid-point anomaly fields, connected-region screening, and
consecutive-day region evolution. Legacy point endpoints remain available.
Ensemble uncertainty, 12 km-to-5 km downscaling, and dynamical physics
validation are unavailable. Thresholds are prototype criteria, not official
warning standards.
"""

import time
import os
import threading
import math
import json
from collections import deque
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================================
# APP CONFIGURATION
# ============================================================

app = FastAPI(
    title="StormTrace Real Forecast & Baseline API",
    description="SIH 26078 Prototype - Real forecast, historical baseline & anomaly service",
    version="0.4.0",
)

DASHBOARD_PATH = Path(__file__).resolve().with_name("weather-tracker.html")
SPATIAL_SNAPSHOT_PATH = Path(__file__).resolve().with_name(".stormtrace_spatial_cache.json")


@app.get("/weather-tracker.html", include_in_schema=False)
@app.get("/", include_in_schema=False)
def dashboard():
    """Serve the standalone StormTrace dashboard from the project directory."""
    return FileResponse(DASHBOARD_PATH, media_type="text/html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Open-Meteo API Endpoints
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_ECMWF_URL = "https://api.open-meteo.com/v1/ecmwf"

DATA_SOURCE_NAME = "Open-Meteo Forecast API"
BASELINE_SOURCE_NAME = "Open-Meteo Historical Weather API (model unspecified in cached values)"
BASELINE_REQUEST_MODEL = "era5"
BASELINE_START_YEAR = 2011
BASELINE_END_YEAR = 2020
BASELINE_PERIOD_NAME = f"{BASELINE_START_YEAR}-{BASELINE_END_YEAR}"
BASELINE_WINDOW_DAYS = 7  # +/- 7 days = 15-day seasonal window around target day-of-year
BASELINE_METHOD_NAME = "seasonally matched historical mean"

# Caching structures
CACHE_TTL_SECONDS = 300  # 5 minutes for live forecast
_FORECAST_CACHE: Dict[str, Dict[str, Any]] = {}
_BASELINE_CACHE: Dict[str, Dict[str, Any]] = {}
_RAW_ARCHIVE_CACHE: Dict[str, Dict[str, Any]] = {}


# Monitored prototype locations across India with real coordinates
LOCATIONS = [
    {
        "id": "odisha-coast",
        "region": "Ganjam, Odisha",
        "latitude": 19.38,
        "longitude": 85.07,
    },
    {
        "id": "bihar",
        "region": "Purnia, Bihar",
        "latitude": 25.78,
        "longitude": 87.47,
    },
    {
        "id": "kanpur",
        "region": "Kanpur Dehat, UP",
        "latitude": 26.42,
        "longitude": 80.35,
    },
    {
        "id": "telangana",
        "region": "Adilabad, Telangana",
        "latitude": 19.67,
        "longitude": 78.53,
    },
]

# ============================================================
# PYDANTIC RESPONSE MODELS
# ============================================================

class AnomalyOut(BaseModel):
    # Core fields required by existing frontend
    id: str
    region: str
    severity: str
    normal_mm: float
    predicted_mm: float
    anomaly_pct: int

    # Real Ingestion Data & Metadata (Task 1)
    latitude: float
    longitude: float
    data_source: str = DATA_SOURCE_NAME
    fetched_at: str
    forecast_10day_mm: Optional[float]

    # Real Historical Baseline & Anomaly Metadata (Task 2)
    baseline_source: str = BASELINE_SOURCE_NAME
    baseline_model_status: Optional[str] = None
    baseline_period: str = BASELINE_PERIOD_NAME
    baseline_method: str = BASELINE_METHOD_NAME
    baseline_sample_days: int = 150
    anomaly_method: str = "percentage anomaly ((predicted - normal) / normal) * 100 (NOT EFI)"

    # Real weather arrays (up to 10 days)
    forecast_timestamps: List[str] = Field(default_factory=list)
    hourly_precipitation: List[Optional[float]] = Field(default_factory=list)
    hourly_precipitation_probability: List[Optional[int]] = Field(default_factory=list)
    hourly_wind_speed: List[Optional[float]] = Field(default_factory=list)
    daily_timestamps: List[str] = Field(default_factory=list)
    daily_precipitation_sum: List[Optional[float]] = Field(default_factory=list)

    # Diagnostic summary flags
    max_hourly_rain_mm: Optional[float] = None
    max_wind_kmh: Optional[float] = None
    max_precip_prob_pct: Optional[int] = None


# ============================================================
# TASK 1: REAL WEATHER FORECAST INGESTION LAYER
# ============================================================

def fetch_real_forecast(location: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetch real 10-day weather forecast from Open-Meteo.
    Uses in-memory caching (5 min TTL) to avoid redundant external calls.
    Handles network errors, timeouts, and API failure gracefully without
    inventing fake fallback values.
    """
    loc_id = location["id"]
    now_ts = time.time()

    # 1. Check in-memory cache
    cached = _FORECAST_CACHE.get(loc_id)
    if cached and (now_ts - cached["cached_at"] < CACHE_TTL_SECONDS):
        return cached

    # 2. Query Open-Meteo Live Forecast API for 10 days
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "hourly": "precipitation,precipitation_probability,wind_speed_10m",
        "daily": "precipitation_sum",
        "forecast_days": 10,
        "timezone": "UTC",
    }

    try:
        response = requests.get(
            OPEN_METEO_FORECAST_URL,
            params=params,
            timeout=10,  # 10s network timeout
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail=f"Open-Meteo Forecast API timed out while fetching data for {location['region']}.",
        )
    except requests.exceptions.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to connect to Open-Meteo Forecast API for {location['region']}: {str(exc)}",
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Open-Meteo Forecast API returned status {response.status_code} for {location['region']}: {response.text}",
        )

    try:
        data = response.json()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Invalid JSON received from Open-Meteo Forecast API for {location['region']}: {str(exc)}",
        )

    # 3. Validate presence of required forecast variables
    if "hourly" not in data or "daily" not in data:
        raise HTTPException(
            status_code=502,
            detail=f"Incomplete forecast payload from Open-Meteo for {location['region']}: missing hourly/daily series",
        )

    fetched_iso = datetime.now(timezone.utc).isoformat()

    entry = {
        "raw": data,
        "cached_at": now_ts,
        "fetched_at": fetched_iso,
        "data_source": DATA_SOURCE_NAME,
    }

    _FORECAST_CACHE[loc_id] = entry
    return entry


def calculate_rainfall_forecast_mm(weather: Dict[str, Any]) -> float:
    """
    Calculate total predicted rainfall over the first 24 hours (REAL DATA).
    Aggregation period: 1 calendar day (24 hours).
    Units: mm
    """
    precipitation = weather.get("hourly", {}).get("precipitation", [])
    first_24 = precipitation[:24]
    if len(first_24) != 24 or any(not isinstance(v, (int, float)) for v in first_24):
        raise HTTPException(status_code=502, detail="Forecast does not contain 24 valid hourly precipitation values.")
    total = sum(first_24)
    return round(total, 1)


def calculate_10_day_rainfall(weather: Dict[str, Any]) -> float:
    """
    Calculate total rainfall across all 10 forecast days (REAL DATA).
    Units: mm
    """
    daily = weather.get("daily", {}).get("precipitation_sum", [])
    if len(daily) < 10 or any(not isinstance(v, (int, float)) for v in daily[:10]):
        raise HTTPException(status_code=502, detail="Forecast does not contain 10 valid daily precipitation totals.")
    total = sum(daily[:10])
    return round(total, 1)


# ============================================================
# TASK 2: REAL HISTORICAL BASELINE & DATA-BASED ANOMALY
# ============================================================

def fetch_historical_baseline(location: Dict[str, Any], ref_date: Optional[date] = None) -> Dict[str, Any]:
    """
    Calculates a climatological baseline using real historical precipitation
    data via Open-Meteo Historical Weather API, explicitly requesting ERA5.

    SCIENTIFIC METHODOLOGY:
    - Target Date: Current forecast reference date (day and month).
    - Seasonal Window: A +/- 7 day seasonal window (15 calendar days centered around
      the target date) across multiple historical years (2011-2020).
    - Sample: 10 years * 15 days = 150 seasonally matched calendar days.
    - Units & Aggregation: Daily precipitation sum (mm).
      This exactly matches `predicted_mm` (first 24-hr daily forecast precipitation).
    - Caching: In-memory cache keyed by location and calendar day, preventing repeated
      network requests.
    - Error Handling: Raises explicit HTTP 502/504 errors on network or API failures.
      NO fake fabricated numbers.
    """
    loc_id = location["id"]
    if ref_date is None:
        ref_date = datetime.now(timezone.utc).date()

    cache_key = f"{loc_id}_{ref_date.month:02d}_{ref_date.day:02d}"

    # 1. Check in-memory baseline cache
    if cache_key in _BASELINE_CACHE:
        return _BASELINE_CACHE[cache_key]

    # 2. Query Open-Meteo Historical Weather API (ERA5 reanalysis)
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "start_date": f"{BASELINE_START_YEAR}-01-01",
        "end_date": f"{BASELINE_END_YEAR}-12-31",
        "daily": "precipitation_sum",
        "models": BASELINE_REQUEST_MODEL,
        "timezone": "UTC",
    }

    # Reuse the explicitly selected ERA5 10-year timeseries if already fetched.
    if loc_id in _RAW_ARCHIVE_CACHE:
        times = _RAW_ARCHIVE_CACHE[loc_id]["times"]
        precip = _RAW_ARCHIVE_CACHE[loc_id]["precip"]
    else:
        try:
            response = requests.get(
                OPEN_METEO_ARCHIVE_URL,
                params=params,
                timeout=15,  # 15s network timeout
            )
        except requests.exceptions.Timeout:
            raise HTTPException(
                status_code=504,
                detail=f"Open-Meteo Historical Weather API timed out for {location['region']}.",
            )
        except requests.exceptions.RequestException as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Failed to connect to Open-Meteo Historical Weather API for {location['region']}: {str(exc)}",
            )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Open-Meteo Historical Weather API returned status {response.status_code} for {location['region']}: {response.text}",
            )

        try:
            data = response.json()
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Invalid JSON received from Historical Weather API for {location['region']}: {str(exc)}",
            )

        daily = data.get("daily", {})
        times = daily.get("time", [])
        precip = daily.get("precipitation_sum", [])
        _RAW_ARCHIVE_CACHE[loc_id] = {"times": times, "precip": precip}

    if not times or not precip or len(times) != len(precip):
        raise HTTPException(
            status_code=502,
            detail=f"Incomplete historical ERA5 precipitation series returned for {location['region']}",
        )

    # 3. Extract seasonally comparable days across 2011-2020
    matched_values: List[float] = []
    for d_str, val in zip(times, precip):
        if val is None or not isinstance(val, (int, float)):
            continue
        try:
            d = date.fromisoformat(d_str)
            # Center window around the equivalent calendar date in that historical year
            target_in_year = date(d.year, ref_date.month, ref_date.day)
            if abs((d - target_in_year).days) <= BASELINE_WINDOW_DAYS:
                matched_values.append(float(val))
        except ValueError:
            # Handles leap day edge case (Feb 29) on non-leap years
            pass

    if not matched_values:
        raise HTTPException(
            status_code=502,
            detail=f"No valid ERA5 precipitation data matched for seasonal window in {location['region']}.",
        )

    # Mean daily historical precipitation for this calendar season (mm)
    normal_mm = round(sum(matched_values) / len(matched_values), 1)

    result = {
        "normal_mm": normal_mm,
        "sample_days": len(matched_values),
        "baseline_source": "Open-Meteo Historical Weather API (ERA5 model)",
        "baseline_model_status": "verified ERA5",
        "baseline_period": BASELINE_PERIOD_NAME,
        "baseline_method": BASELINE_METHOD_NAME,
        "calculated_at": datetime.now(timezone.utc).isoformat(),
    }

    _BASELINE_CACHE[cache_key] = result
    return result


def calculate_anomaly(predicted_mm: float, normal_mm: float) -> int:
    """
    Calculates the rainfall percentage anomaly:
        anomaly_pct = ((predicted_mm - normal_mm) / normal_mm) * 100

    REAL / DATA-BASED METRIC.
    Handles zero or extremely small historical baselines safely.
    NOTE: This is a percentage anomaly, NOT an Extreme Forecast Index (EFI).
    """
    # Safe handling of zero or near-zero historical baseline
    if normal_mm <= 0.05:
        if predicted_mm <= 0.05:
            return 0
        # When normal is near-zero, compute percentage relative to 0.1 mm reference
        return round(((predicted_mm - 0.1) / 0.1) * 100)

    pct = ((predicted_mm - normal_mm) / normal_mm) * 100
    return round(pct)


def classify_severity(anomaly_pct: int) -> str:
    """
    Prototype severity classification based on percentage anomaly.
    [DEMO THRESHOLDS ONLY]
    NOTE: These thresholds are illustrative demo thresholds for the SIH prototype
    and are NOT official IMD (India Meteorological Department) warning thresholds.
    """
    if anomaly_pct >= 50:
        return "severe"
    if anomaly_pct >= 25:
        return "moderate"
    return "low"


# ============================================================
# BUILD COMPLETE REAL ANOMALY OBJECT
# ============================================================

def build_anomaly(location: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fetches real forecast data from Open-Meteo, computes the real ERA5
    historical baseline for the matching calendar period, and builds the
    unified anomaly response with 100% frontend compatibility.
    """
    # 1. Fetch real forecast (or retrieve from in-memory cache)
    try:
        cache_entry = fetch_real_forecast(location)
    except HTTPException:
        # A real spatial ECMWF+ERA5 result remains a valid compatibility
        # source when a point endpoint is temporarily rate-limited.
        return _spatial_compat_anomaly(location)
    raw_weather = cache_entry["raw"]
    fetched_at = cache_entry["fetched_at"]

    # 2. Extract real forecast metrics
    predicted_mm = calculate_rainfall_forecast_mm(raw_weather)
    forecast_10day_mm = calculate_10_day_rainfall(raw_weather)

    hourly = raw_weather.get("hourly", {})
    daily = raw_weather.get("daily", {})

    hourly_precip = [float(x) if isinstance(x, (int, float)) else None for x in hourly.get("precipitation", [])]
    hourly_prob = hourly.get("precipitation_probability", [])
    hourly_wind = [float(x) if isinstance(x, (int, float)) else None for x in hourly.get("wind_speed_10m", [])]

    daily_sum = [float(x) if isinstance(x, (int, float)) else None for x in daily.get("precipitation_sum", [])]
    daily_times = daily.get("time", [])
    forecast_timestamps = hourly.get("time", [])

    valid_hourly_rain = [x for x in hourly_precip if x is not None]
    max_hourly_rain = max(valid_hourly_rain) if valid_hourly_rain else None
    valid_winds = [x for x in hourly_wind if x is not None]
    max_wind = max(valid_winds) if valid_winds else None
    valid_probs = [x for x in hourly_prob if isinstance(x, (int, float))]
    max_prob = max(valid_probs) if valid_probs else None

    # 3. Compute REAL ERA5 historical baseline for the matched seasonal window
    try:
        baseline_info = fetch_historical_baseline(location)
    except HTTPException:
        return _spatial_compat_anomaly(location)
    normal_mm = baseline_info["normal_mm"]

    # 4. Calculate real data-based anomaly percentage
    anomaly_pct = calculate_anomaly(predicted_mm, normal_mm)
    severity = classify_severity(anomaly_pct)

    return {
        # Core fields consumed by weather-tracker.html
        "id": location["id"],
        "region": location["region"],
        "severity": severity,
        "normal_mm": normal_mm,
        "predicted_mm": predicted_mm,
        "anomaly_pct": anomaly_pct,

        # Real Ingestion Data & Metadata (Task 1)
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "data_source": DATA_SOURCE_NAME,
        "fetched_at": fetched_at,
        "forecast_10day_mm": forecast_10day_mm,

        # Real Historical Baseline & Anomaly Metadata (Task 2)
        "baseline_source": baseline_info["baseline_source"],
        "baseline_model_status": baseline_info["baseline_model_status"],
        "baseline_period": baseline_info["baseline_period"],
        "baseline_method": baseline_info["baseline_method"],
        "baseline_sample_days": baseline_info["sample_days"],
        "anomaly_method": "percentage anomaly ((predicted - normal) / normal) * 100 (NOT EFI)",

        # Detailed Real Time-Series (10 Days)
        "forecast_timestamps": forecast_timestamps,
        "hourly_precipitation": hourly_precip,
        "hourly_precipitation_probability": hourly_prob,
        "hourly_wind_speed": hourly_wind,
        "daily_timestamps": daily_times,
        "daily_precipitation_sum": daily_sum,

        # Real diagnostic metrics
        "max_hourly_rain_mm": round(max_hourly_rain, 1) if max_hourly_rain is not None else None,
        "max_wind_kmh": round(max_wind, 1) if max_wind is not None else None,
        "max_precip_prob_pct": int(max_prob) if max_prob is not None else None,
    }


def get_all_anomalies() -> List[Dict[str, Any]]:
    """
    Fetch real forecast data and real ERA5 baselines for all monitored locations.
    Uses ThreadPoolExecutor for concurrent API requests to ensure responsive loading.
    Gracefully aggregates errors if any location fails without inventing fake numbers.
    """
    results_map: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []

    def _fetch_loc(loc: Dict[str, Any]):
        try:
            return loc["id"], build_anomaly(loc), None
        except Exception as exc:
            return loc["id"], None, f"{loc['region']}: {str(exc)}"

    with ThreadPoolExecutor(max_workers=len(LOCATIONS)) as executor:
        futures = [executor.submit(_fetch_loc, loc) for loc in LOCATIONS]
        for f in as_completed(futures):
            loc_id, res, err = f.result()
            if res:
                results_map[loc_id] = res
            if err:
                errors.append(err)
                print(f"[StormTrace] Error processing {loc_id}: {err}")

    # Maintain original order of LOCATIONS
    results = [results_map[loc["id"]] for loc in LOCATIONS if loc["id"] in results_map]

    if not results and errors:
        raise HTTPException(
            status_code=503,
            detail=f"Live forecast / baseline ingestion failed for all monitored locations: {'; '.join(errors)}"
        )

    return results


def _find_anomaly(anomaly_id: str) -> Dict[str, Any]:
    """
    Find specific anomaly by ID and populate with real forecast & baseline data.
    """
    for location in LOCATIONS:
        if location["id"] == anomaly_id:
            return build_anomaly(location)

    raise HTTPException(
        status_code=404,
        detail=f"Unknown anomaly_id '{anomaly_id}'. Valid IDs: {[loc['id'] for loc in LOCATIONS]}",
    )


# ============================================================
# TASK 3: REAL 3-10 DAY ANOMALY TRACKING & EVENT SUMMARY
# ============================================================

def get_tracking_data(anomaly_id: str) -> Dict[str, Any]:
    """
    Task 3: Real 3-10 Day Anomaly Tracking & Temporal Evolution.
    Replaces hardcoded simulated day offsets with data-driven temporal tracking
    across all 10 forecast days using live Open-Meteo forecast and ERA5 baselines.

    REAL vs SIMULATED:
    - Forecast rainfall, wind speed, precip probability: REAL (Open-Meteo)
    - Historical baseline (daily matched): REAL (ERA5 2011-2020)
    - Daily anomaly percentage & severity: REAL / DATA-BASED
    - Event summary (peak, persistence, max rain): REAL / DYNAMICALLY CALCULATED
    - Spatial motion is not inferred; all daily values refer to the same forecast point.
    """
    a = _find_anomaly(anomaly_id)
    if a.get("spatial_grid_fallback"):
        return _spatial_compat_tracking(a)
    loc = next((l for l in LOCATIONS if l["id"] == anomaly_id), None)
    if not loc:
        raise HTTPException(status_code=404, detail=f"Location {anomaly_id} not found")

    daily_times = a.get("daily_timestamps", [])
    daily_rain = a.get("daily_precipitation_sum", [])
    hourly_prob = a.get("hourly_precipitation_probability", [])
    hourly_wind = a.get("hourly_wind_speed", [])

    days_list = []
    for i in range(len(daily_times)):
        day_num = i + 1
        date_str = daily_times[i]
        rain_mm = round(float(daily_rain[i]), 1) if i < len(daily_rain) and isinstance(daily_rain[i], (int, float)) else None

        # ERA5 baseline for this exact calendar date
        target_date = date.fromisoformat(date_str)
        base_info = fetch_historical_baseline(loc, target_date)
        baseline_mm = base_info["normal_mm"]

        # Data-based percentage anomaly & severity
        anomaly_pct = calculate_anomaly(rain_mm, baseline_mm) if rain_mm is not None else None
        severity = classify_severity(anomaly_pct) if anomaly_pct is not None else "unavailable"

        # Hourly aggregates for that day (24 hours per day)
        day_probs = [p for p in hourly_prob[i*24 : (i+1)*24] if isinstance(p, (int, float))]
        max_prob = int(max(day_probs)) if day_probs else None

        day_winds = [w for w in hourly_wind[i*24 : (i+1)*24] if isinstance(w, (int, float))]
        max_wind = round(float(max(day_winds)), 1) if day_winds else None
        # Data-responsive screening radius for map emphasis only; it is not an
        # observed storm boundary or validated impact footprint.
        radius_terms = [rain_mm * 0.45] if rain_mm is not None else None
        if radius_terms is not None and max_prob is not None:
            radius_terms.append(max_prob * 0.18)
        screening_radius = round(min(80.0, 12.0 + sum(radius_terms)), 1) if radius_terms is not None else None

        days_list.append({
            "day": day_num,
            "forecast_date": date_str,
            "rainfall_mm": rain_mm,
            "baseline_mm": baseline_mm,
            "anomaly_pct": anomaly_pct,
            "severity": severity,
            "precip_probability_pct": max_prob,
            "wind_speed_kmh": max_wind,

            "latitude": loc["latitude"],
            "longitude": loc["longitude"],
            "screening_radius_km": screening_radius,
            "footprint_status": "heuristic screening radius around a fixed forecast point; not a storm boundary",
            "spatial_status": "fixed forecast point; storm motion is not inferred",
        })

    # Event summary calculations
    valid_anomaly_days = [d for d in days_list if d["anomaly_pct"] is not None]
    valid_rain_days = [d for d in days_list if d["rainfall_mm"] is not None]
    peak_entry = max(valid_anomaly_days, key=lambda d: d["anomaly_pct"]) if valid_anomaly_days else None
    max_rain_entry = max(valid_rain_days, key=lambda d: d["rainfall_mm"]) if valid_rain_days else None

    # Qualifying days for persistence (prototype demo threshold: anomaly_pct >= 25)
    anomalous_days = [d["day"] for d in valid_anomaly_days if d["anomaly_pct"] >= 25]
    persistence_days = len(anomalous_days)
    first_anomalous = anomalous_days[0] if anomalous_days else None
    last_anomalous = anomalous_days[-1] if anomalous_days else None

    event_summary = {
        "peak_anomaly_pct": peak_entry["anomaly_pct"] if peak_entry else None,
        "peak_day": peak_entry["day"] if peak_entry else None,
        "peak_forecast_date": peak_entry["forecast_date"] if peak_entry else "",
        "persistence_days": persistence_days,
        "first_anomalous_day": first_anomalous,
        "last_anomalous_day": last_anomalous,
        "max_rainfall_mm": max_rain_entry["rainfall_mm"] if max_rain_entry else None,
        "max_rainfall_day": max_rain_entry["day"] if max_rain_entry else None,
        "thresholds_note": "Severity thresholds (>=50% severe, >=25% moderate) are illustrative prototype demonstration thresholds, NOT official IMD warning thresholds."
    }

    return {
        "region": a["region"],
        "tracking_mode": "REAL FORECAST TIME-SERIES ANALYSIS",
        "forecast_source": DATA_SOURCE_NAME,
        "baseline_source": BASELINE_SOURCE_NAME,
        "spatial_tracking_status": "not implemented; daily values refer to one fixed forecast point",
        "days": days_list,
        "event_summary": event_summary,
    }


def downscale_status(anomaly_id: str):
    _find_anomaly(anomaly_id)
    return {
        "status": "unavailable",
        "method_status": "pending real gridded NWP input and validated localization method",
        "reason": "The current forecast source provides point forecasts here; no 12 km grid is available to downscale.",
        "coarse": None,
        "fine": None,
    }


def uncertainty_status(anomaly_id: str):
    _find_anomaly(anomaly_id)
    return {
        "ensemble_status": "unavailable",
        "reason": "No ensemble member data is connected. A confidence percentage or probability would be unsupported.",
        "uncertainty_source": None,
        "members": None,
    }


def check_physical_consistency(anomaly: Dict[str, Any]) -> Dict[str, Any]:
    """Basic data/aggregation checks, not a dynamical atmospheric physics model."""
    hourly = anomaly.get("hourly_precipitation", [])
    daily = anomaly.get("daily_precipitation_sum", [])
    checks = []
    valid_hourly = [v for v in hourly if isinstance(v, (int, float))]
    valid_daily = [v for v in daily if isinstance(v, (int, float))]
    checks.append({"name": "non-negative hourly precipitation", "status": "pass" if valid_hourly and all(v >= 0 for v in valid_hourly) else "fail" if valid_hourly else "unavailable"})
    valid_probs = [v for v in anomaly.get("hourly_precipitation_probability", []) if isinstance(v, (int, float))]
    checks.append({"name": "precipitation probability within 0-100%", "status": "pass" if valid_probs and all(0 <= v <= 100 for v in valid_probs) else "fail" if valid_probs else "unavailable"})
    valid_winds = [v for v in anomaly.get("hourly_wind_speed", []) if isinstance(v, (int, float))]
    checks.append({"name": "non-negative 10 m wind speed", "status": "pass" if valid_winds and all(v >= 0 for v in valid_winds) else "fail" if valid_winds else "unavailable"})
    comparable = min(len(daily), len(hourly) // 24)
    if comparable:
        differences = []
        for i in range(comparable):
            block = hourly[i * 24:(i + 1) * 24]
            if all(isinstance(v, (int, float)) for v in block) and isinstance(daily[i], (int, float)):
                differences.append(abs(sum(block) - daily[i]))
        checks.append({"name": "daily total agrees with hourly accumulation (<=0.2 mm)", "status": "pass" if differences and max(differences) <= 0.2 else "fail" if differences else "unavailable"})
    return {"status": "basic_consistency_checks", "scope": "range and aggregation checks only; no atmospheric dynamics validation", "checks": checks}


# ============================================================
# API ROUTES
# ============================================================

# ============================================================
# REAL SPATIAL ECMWF + ERA5 PIPELINE
# ============================================================

GRID_MIN_LAT = float(os.getenv("STORMTRACE_MIN_LAT", "6"))
GRID_MAX_LAT = float(os.getenv("STORMTRACE_MAX_LAT", "36"))
GRID_MIN_LON = float(os.getenv("STORMTRACE_MIN_LON", "66"))
GRID_MAX_LON = float(os.getenv("STORMTRACE_MAX_LON", "100"))
GRID_SPACING = float(os.getenv("STORMTRACE_GRID_SPACING", "3"))
GRID_BATCH_SIZE = max(1, min(48, int(os.getenv("STORMTRACE_BATCH_SIZE", "48"))))
# ECMWF IFS updates every six hours. A three-hour application TTL limits repeat
# provider requests while still refreshing halfway between model cycles.
GRID_CACHE_TTL = 3 * 60 * 60
ANOMALY_THRESHOLD_PCT = float(os.getenv("STORMTRACE_ANOMALY_THRESHOLD_PCT", "50"))
MIN_EXTREME_RAIN_MM = float(os.getenv("STORMTRACE_MIN_EXTREME_RAIN_MM", "10"))
MIN_REGION_CELLS = max(1, int(os.getenv("STORMTRACE_MIN_REGION_CELLS", "2")))
_SPATIAL_CACHE: Dict[str, Any] = {}
_SPATIAL_LOCK = threading.RLock()


def _make_grid() -> List[Dict[str, Any]]:
    if GRID_SPACING <= 0 or GRID_MAX_LAT < GRID_MIN_LAT or GRID_MAX_LON < GRID_MIN_LON:
        raise RuntimeError("Invalid StormTrace spatial grid configuration")
    cells = []
    row = 0
    lat = GRID_MIN_LAT
    while lat <= GRID_MAX_LAT + 1e-8:
        col = 0
        lon = GRID_MIN_LON
        while lon <= GRID_MAX_LON + 1e-8:
            cells.append({"id": f"g{row:02d}-{col:02d}", "row": row, "col": col,
                          "latitude": round(lat, 5), "longitude": round(lon, 5)})
            col += 1
            lon += GRID_SPACING
        row += 1
        lat += GRID_SPACING
    return cells


SPATIAL_GRID = _make_grid()


def _get_batch(url: str, params: Dict[str, Any]) -> Any:
    last_error = None
    # At most one retry. In particular, do not keep retrying a daily-quota 429.
    for attempt in range(2):
        try:
            response = requests.get(url, params=params, timeout=45)
            if response.status_code == 429:
                body = response.text.lower()
                if "daily api request limit" in body:
                    raise RuntimeError(f"Open-Meteo HTTP 429: {response.text[:500]}")
                if attempt == 0:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after else 60.0 if "one minute" in body else 60.0
                    except ValueError:
                        delay = 60.0
                    time.sleep(max(1.0, min(delay, 65.0)))
                    continue
                raise RuntimeError(f"Open-Meteo HTTP 429: {response.text[:500]}")
            if response.status_code in (500, 502, 503, 504) and attempt == 0:
                time.sleep(2.0)
                continue
            if response.status_code != 200:
                raise RuntimeError(f"Open-Meteo HTTP {response.status_code}: {response.text[:500]}")
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == 0:
                time.sleep(2.0)
    raise RuntimeError(str(last_error or "Open-Meteo request failed"))


def _coordinate_batches() -> List[List[Dict[str, Any]]]:
    return [SPATIAL_GRID[i:i + GRID_BATCH_SIZE] for i in range(0, len(SPATIAL_GRID), GRID_BATCH_SIZE)]


def _baseline_values_from_snapshot(snapshot: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Reuse historical summaries while retaining model provenance when known."""
    cached: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not snapshot:
        return cached
    for day in snapshot.get("days", []):
        forecast_date = day.get("forecast_date")
        if not forecast_date:
            continue
        for cell in day.get("cells", []):
            if _finite_number(cell.get("baseline_mean_mm")) is None:
                continue
            cached.setdefault(cell["id"], {})[forecast_date] = {
                "mean_mm": cell.get("baseline_mean_mm"),
                "std_mm": cell.get("baseline_std_mm"),
                "sample_days": cell.get("baseline_sample_days", 0),
                "model": cell.get("baseline_model", "unspecified"),
            }
    return cached


def _run_spatial_batches(batches: List[List[Dict[str, Any]]], historical: bool,
                         gathered: Dict[str, Dict[str, Any]], errors: List[str]) -> None:
    """Run at most two multi-coordinate requests at once and keep per-cell failures explicit."""
    def request_batch(batch: List[Dict[str, Any]]):
        common = {"latitude": ",".join(str(c["latitude"]) for c in batch),
                  "longitude": ",".join(str(c["longitude"]) for c in batch), "timezone": "UTC"}
        if historical:
            params = {**common, "start_date": f"{BASELINE_START_YEAR}-01-01",
                      "end_date": f"{BASELINE_END_YEAR}-12-31", "daily": "precipitation_sum",
                      "models": BASELINE_REQUEST_MODEL}
            payload = _get_batch(OPEN_METEO_ARCHIVE_URL, params)
        else:
            params = {**common, "hourly": "precipitation,wind_speed_10m,temperature_2m",
                      "forecast_days": 10, "wind_speed_unit": "kmh", "cell_selection": "nearest"}
            payload = _get_batch(OPEN_METEO_ECMWF_URL, params)
        return batch, payload if isinstance(payload, list) else [payload]

    label = "ERA5" if historical else "ECMWF"
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(request_batch, batch): batch for batch in batches}
        for future in as_completed(futures):
            batch = futures[future]
            try:
                _, payloads = future.result()
                for index, coord in enumerate(batch):
                    if index >= len(payloads) or not isinstance(payloads[index], dict):
                        errors.append(f"{label} batch missing {coord['id']}")
                        continue
                    gathered[coord["id"]]["history" if historical else "forecast"] = payloads[index]
            except Exception as exc:
                errors.append(f"{label} batch failed ({len(batch)} coordinates): {exc}")


def _fetch_spatial_inputs() -> Dict[str, Any]:
    now = time.time()
    cached = _SPATIAL_CACHE.get("analysis")
    if cached and now - cached["cached_at"] < GRID_CACHE_TTL:
        value = cached["value"]
        age = int(value.get("cache_age_seconds", 0) + now - cached["cached_at"])
        delivered = {**value, "freshness_status": "cached", "cache_age_seconds": max(0, age)}
        cached["last_delivery_status"], cached["last_delivery_at"] = "cached", now
        return delivered
    with _SPATIAL_LOCK:
        now = time.time()
        cached = _SPATIAL_CACHE.get("analysis")
        if cached and now - cached["cached_at"] < GRID_CACHE_TTL:
            value = cached["value"]
            age = int(value.get("cache_age_seconds", 0) + now - cached["cached_at"])
            delivered = {**value, "freshness_status": "cached", "cache_age_seconds": max(0, age)}
            cached["last_delivery_status"], cached["last_delivery_at"] = "cached", now
            return delivered
        saved = _load_spatial_snapshot()
        # Reuse the on-disk snapshot only within the normal cache TTL. Once it
        # expires, keep it as fallback while requesting a fresh forecast.
        stale_snapshot = saved
        if saved and now - float(saved.get("snapshot_age_seconds", 0)) < GRID_CACHE_TTL:
            _SPATIAL_CACHE["analysis"] = {"cached_at": now, "value": saved,
                                           "last_delivery_status": "cached", "last_delivery_at": now}
            return {**saved, "freshness_status": "cached",
                    "cache_age_seconds": max(0, int(saved.get("snapshot_age_seconds", 0)))}
        batches = _coordinate_batches()
        gathered: Dict[str, Dict[str, Any]] = {c["id"]: {**c, "forecast": None, "history": None} for c in SPATIAL_GRID}
        errors = []
        _run_spatial_batches(batches, False, gathered, errors)

        valid_forecasts = sum(1 for item in gathered.values() if isinstance((item.get("forecast") or {}).get("hourly"), dict))
        if valid_forecasts == 0:
            if stale_snapshot:
                stale = _mark_spatial_snapshot_stale(stale_snapshot, "Live ECMWF batches unavailable; serving the last real cached field.")
                _SPATIAL_CACHE["analysis"] = {"cached_at": now, "value": stale,
                                               "last_delivery_status": "cached", "last_delivery_at": now}
                return stale
            raise HTTPException(status_code=503, detail="ECMWF spatial forecast is unavailable; no valid grid points were returned.")
        forecast_dates = sorted({stamp[:10] for item in gathered.values()
                                 for stamp in ((item.get("forecast") or {}).get("hourly") or {}).get("time", [])})[:10]
        cached_baselines = _baseline_values_from_snapshot(stale_snapshot)
        baseline_retry_after_epoch = None

        # Historical baseline is stable for the already cached target dates. Request the
        # large 2011-2020 archive only for coordinates/dates not covered by that cache.
        missing_history = [coord for coord in SPATIAL_GRID
                           if not all(d in cached_baselines.get(coord["id"], {}) for d in forecast_dates)]
        if stale_snapshot:
            retry_after = stale_snapshot.get("baseline_retry_after_epoch")
            daily_limit_seen = any("daily api request limit exceeded" in str(error).lower()
                                   for error in stale_snapshot.get("errors", []))
            if daily_limit_seen and not retry_after:
                try:
                    retry_after = SPATIAL_SNAPSHOT_PATH.stat().st_mtime + 24 * 60 * 60
                except OSError:
                    retry_after = now + 24 * 60 * 60
            if retry_after and now < float(retry_after):
                baseline_retry_after_epoch = float(retry_after)
                errors.append("ERA5 archive refresh deferred after provider daily quota response; existing baselines are reused where available.")
                missing_history = []
        if missing_history:
            history_batches = [missing_history[i:i + GRID_BATCH_SIZE]
                               for i in range(0, len(missing_history), GRID_BATCH_SIZE)]
            _run_spatial_batches(history_batches, True, gathered, errors)
        try:
            analysis = _build_spatial_analysis(gathered, errors, cached_baselines)
        except HTTPException as exc:
            if stale_snapshot and exc.status_code >= 500:
                stale = _mark_spatial_snapshot_stale(stale_snapshot, f"Live ECMWF/ERA5 responses were unusable; serving the last real cached field. ({exc.detail})")
                _SPATIAL_CACHE["analysis"] = {"cached_at": now, "value": stale,
                                               "last_delivery_status": "cached", "last_delivery_at": now}
                return stale
            raise
        if not any(cell["data_valid"] for day in analysis["days"] for cell in day["cells"]):
            if stale_snapshot:
                stale = _mark_spatial_snapshot_stale(stale_snapshot, "Live responses contained no complete ECMWF/ERA5 grid cells; serving the last real cached field.")
                _SPATIAL_CACHE["analysis"] = {"cached_at": now, "value": stale,
                                               "last_delivery_status": "cached", "last_delivery_at": now}
                return stale
            raise HTTPException(status_code=503, detail="ECMWF/ERA5 returned no grid points with both valid forecast and baseline values.")
        if any("daily api request limit exceeded" in str(error).lower() for error in errors):
            baseline_retry_after_epoch = time.time() + 24 * 60 * 60
        if baseline_retry_after_epoch:
            analysis["baseline_retry_after_epoch"] = baseline_retry_after_epoch
        fetched_at = time.time()
        _SPATIAL_CACHE["analysis"] = {"cached_at": fetched_at, "value": analysis,
                                       "last_delivery_status": "live", "last_delivery_at": fetched_at}
        _save_spatial_snapshot(analysis)
        return analysis


def _save_spatial_snapshot(analysis: Dict[str, Any]) -> None:
    """Persist real source output briefly so restarts don't discard today's fetch."""
    try:
        payload = {"saved_at": datetime.now(timezone.utc).isoformat(), "analysis": analysis}
        temporary = SPATIAL_SNAPSHOT_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        temporary.replace(SPATIAL_SNAPSHOT_PATH)
    except OSError as exc:
        print(f"[StormTrace] Could not persist spatial cache: {exc}")


def _load_spatial_snapshot() -> Optional[Dict[str, Any]]:
    try:
        if not SPATIAL_SNAPSHOT_PATH.exists():
            return None
        age_seconds = time.time() - SPATIAL_SNAPSHOT_PATH.stat().st_mtime
        payload = json.loads(SPATIAL_SNAPSHOT_PATH.read_text(encoding="utf-8-sig"))
        if isinstance(payload.get("analysis"), dict):
            result = payload["analysis"]
            coverage = result.get("coverage_status") or ("partial" if result.get("data_status") == "partial" else "complete")
            saved_at = datetime.fromtimestamp(SPATIAL_SNAPSHOT_PATH.stat().st_mtime, timezone.utc).isoformat()
            first_cells = result.get("days", [{}])[0].get("cells", []) if result.get("days") else []
            # Snapshots created before the Archive API request included an
            # explicit model selector have unverified historical provenance.
            for day in result.get("days", []):
                for cell in day.get("cells", []):
                    if cell.get("baseline_mean_mm") is not None:
                        cell.setdefault("baseline_model", "unspecified")
            grid = {**result.get("grid", {}),
                    "valid_forecast_points": sum(c.get("precipitation_mm") is not None for c in first_cells),
                    "valid_baseline_points": sum(c.get("baseline_mean_mm") is not None for c in first_cells),
                    "valid_analysis_points": sum(bool(c.get("data_valid")) for c in first_cells)}
            baseline_models = {c.get("baseline_model", "unavailable") for d in result.get("days", [])
                               for c in d.get("cells", []) if c.get("baseline_mean_mm") is not None}
            model_status = ("unavailable" if not baseline_models else
                            "verified ERA5" if baseline_models == {"era5"} else
                            "unverified; model unspecified in cached values" if baseline_models <= {"unspecified"} else
                            "mixed; includes cached values with unspecified model")
            result = {**result, "coverage_status": coverage, "freshness_status": "cached",
                      "baseline_model_status": model_status,
                      "baseline_source": ("Open-Meteo Historical Weather API (ERA5 model)"
                                          if model_status == "verified ERA5" else BASELINE_SOURCE_NAME),
                      "grid": grid,
                      "cache_age_seconds": int(age_seconds),
                      "snapshot_age_seconds": age_seconds,
                      "forecast_retrieved_at": result.get("forecast_retrieved_at") or saved_at,
                      "data_status": "stale" if age_seconds >= GRID_CACHE_TTL else coverage,
                      "stale_age_minutes": round(age_seconds/60, 1) if age_seconds >= GRID_CACHE_TTL else None}
            return result
        fields = payload.get("fields")
        if not isinstance(fields, list) or len(fields) != 10:
            return None
        by_id = {c["id"]: c for c in SPATIAL_GRID}
        days, regions_by_day, evolution = [], [], []
        for field in fields:
            valid = {f["properties"]["id"]: f["properties"] for f in field.get("features", [])}
            cells = []
            for cid, base in by_id.items():
                p = valid.get(cid)
                if p is None:
                    cells.append({**base, "forecast_date": field["forecast_date"], "precipitation_mm": None,
                                  "baseline_mean_mm": None, "baseline_std_mm": None, "baseline_sample_days": 0,
                                  "anomaly_pct": None, "extreme_anomaly_score_z": None,
                                  "wind_speed_max_kmh": None, "temperature_mean_c": None,
                                  "anomaly_mask": False, "data_valid": False})
                else:
                    cells.append({**base, **p, "data_valid": p.get("precipitation_mm") is not None and p.get("baseline_mean_mm") is not None})
            day = {"day": field["day"], "forecast_date": field["forecast_date"], "cells": cells}
            days.append(day)
            regions = field.get("regions", [])
            day_events = field.get("evolution", [])
            regions_by_day.append({"day": field["day"], "forecast_date": field["forecast_date"],
                                   "regions": regions, "evolution": day_events})
            evolution.extend(day_events)
        first = fields[0]
        original_status = first.get("data_status", "partial")
        coverage = original_status if original_status in ("partial", "complete") else "partial"
        saved_at = datetime.fromtimestamp(SPATIAL_SNAPSHOT_PATH.stat().st_mtime, timezone.utc).isoformat()
        return {"created_at": payload.get("saved_at", first.get("forecast_date")),
                "coverage_status": coverage, "freshness_status": "cached", "baseline_freshness": "cached",
                "cache_age_seconds": int(age_seconds), "forecast_retrieved_at": saved_at,
                "forecast_start_time": f"{fields[0].get('forecast_date')}T00:00",
                "forecast_end_time": f"{fields[-1].get('forecast_date')}T23:00",
                "data_status": "stale" if age_seconds >= GRID_CACHE_TTL else coverage,
                "stale_age_minutes": round(age_seconds/60, 1) if age_seconds >= GRID_CACHE_TTL else None,
                "errors": first.get("errors", []),
                "forecast_source": first.get("forecast_source", "ECMWF IFS via Open-Meteo ECMWF API"),
                "baseline_source": first.get("baseline_source", BASELINE_SOURCE_NAME),
                "baseline_period": first.get("baseline_period", BASELINE_PERIOD_NAME),
                "grid": first.get("grid", {}), "variables": first.get("variables", []),
                "unavailable_variables": first.get("unavailable_variables", []),
                "method": first.get("method", {}), "days": days,
                "regions_by_day": regions_by_day, "evolution": evolution}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"[StormTrace] Ignoring unreadable spatial cache: {exc}")
        return None


def _mark_spatial_snapshot_stale(analysis: Dict[str, Any], reason: str) -> Dict[str, Any]:
    try:
        age = max(0, time.time() - SPATIAL_SNAPSHOT_PATH.stat().st_mtime)
    except OSError:
        age = 0
    return {**analysis, "data_status": "stale", "coverage_status": analysis.get("coverage_status", "partial"),
            "freshness_status": "cached", "cache_age_seconds": int(age),
            "stale_age_minutes": round(age/60, 1),
            "errors": (analysis.get("errors", []) + [reason])[:20]}


def _finite_number(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def _seasonal_baseline(history: Dict[str, Any], target: date) -> Dict[str, Any]:
    daily = history.get("daily") if isinstance(history, dict) else None
    times = daily.get("time", []) if isinstance(daily, dict) else []
    values = daily.get("precipitation_sum", []) if isinstance(daily, dict) else []
    matched = []
    for d_str, value in zip(times, values):
        n = _finite_number(value)
        if n is None:
            continue
        try:
            observed = date.fromisoformat(d_str)
            seasonal_day = date(observed.year, target.month, target.day)
            if abs((observed - seasonal_day).days) <= BASELINE_WINDOW_DAYS:
                matched.append(n)
        except (ValueError, TypeError):
            continue
    if not matched:
        return {"mean_mm": None, "std_mm": None, "sample_days": 0}
    mean = sum(matched) / len(matched)
    variance = sum((x - mean) ** 2 for x in matched) / len(matched)
    return {"mean_mm": round(mean, 2), "std_mm": round(math.sqrt(variance), 2), "sample_days": len(matched)}


def _build_spatial_analysis(gathered: Dict[str, Dict[str, Any]], errors: List[str],
                            cached_baselines: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None) -> Dict[str, Any]:
    cached_baselines = cached_baselines or {}
    dates = []
    forecast_timestamps = set()
    for item in gathered.values():
        hourly = item.get("forecast", {}).get("hourly", {})
        for stamp in hourly.get("time", []):
            forecast_timestamps.add(stamp)
            day_text = stamp[:10]
            if day_text not in dates:
                dates.append(day_text)
        if len(dates) >= 10:
            break
    dates = dates[:10]
    if not dates:
        raise HTTPException(status_code=503, detail="ECMWF forecast returned no usable timestamps.")

    all_days = []
    baseline_origins = set()
    for day_index, day_text in enumerate(dates):
        forecast_date = date.fromisoformat(day_text)
        cells = []
        for item in gathered.values():
            fc = item.get("forecast") or {}
            hourly = fc.get("hourly") or {}
            times = hourly.get("time", [])
            indices = [i for i, stamp in enumerate(times) if stamp.startswith(day_text)]
            p = hourly.get("precipitation", [])
            w = hourly.get("wind_speed_10m", [])
            t = hourly.get("temperature_2m", [])
            rain_values = [_finite_number(p[i]) for i in indices if i < len(p)]
            wind_values = [_finite_number(w[i]) for i in indices if i < len(w)]
            temp_values = [_finite_number(t[i]) for i in indices if i < len(t)]
            rain_values = [x for x in rain_values if x is not None]
            wind_values = [x for x in wind_values if x is not None]
            temp_values = [x for x in temp_values if x is not None]
            rain = round(sum(rain_values), 2) if len(rain_values) == 24 else None
            baseline = _seasonal_baseline(item.get("history") or {}, forecast_date)
            baseline_model = "era5"
            if baseline["mean_mm"] is not None:
                baseline_origins.add("live")
            else:
                baseline = cached_baselines.get(item["id"], {}).get(day_text, baseline)
                if baseline.get("mean_mm") is not None:
                    baseline_origins.add("cached")
                    baseline_model = baseline.get("model", "unspecified")
                else:
                    baseline_model = "unavailable"
            normal = baseline["mean_mm"]
            anomaly = calculate_anomaly(rain, normal) if rain is not None and normal is not None else None
            std = baseline["std_mm"]
            zscore = round((rain - normal) / std, 2) if rain is not None and normal is not None and std is not None and std >= 0.1 else None
            mask = bool(anomaly is not None and anomaly >= ANOMALY_THRESHOLD_PCT and rain is not None and rain >= MIN_EXTREME_RAIN_MM)
            cells.append({**{k: item[k] for k in ("id", "row", "col", "latitude", "longitude")},
                          "forecast_date": day_text, "precipitation_mm": rain,
                          "baseline_mean_mm": normal, "baseline_std_mm": std,
                          "baseline_sample_days": baseline["sample_days"], "baseline_model": baseline_model,
                          "anomaly_pct": anomaly,
                          "extreme_anomaly_score_z": zscore,
                          "wind_speed_max_kmh": round(max(wind_values), 1) if wind_values else None,
                          "temperature_mean_c": round(sum(temp_values) / len(temp_values), 1) if temp_values else None,
                          "anomaly_mask": mask, "data_valid": rain is not None and normal is not None})
        all_days.append({"day": day_index + 1, "forecast_date": day_text, "cells": cells})

    regions_by_day = []
    for day in all_days:
        cells = {c["id"]: c for c in day["cells"]}
        active = {c["id"] for c in day["cells"] if c["anomaly_mask"]}
        components = []
        while active:
            seed = active.pop()
            q = deque([seed]); component = [seed]
            while q:
                current = cells[q.popleft()]
                for neighbor_id in (f"g{current['row']-1:02d}-{current['col']:02d}", f"g{current['row']+1:02d}-{current['col']:02d}",
                                    f"g{current['row']:02d}-{current['col']-1:02d}", f"g{current['row']:02d}-{current['col']+1:02d}"):
                    if neighbor_id in active:
                        active.remove(neighbor_id); q.append(neighbor_id); component.append(neighbor_id)
            if len(component) >= MIN_REGION_CELLS:
                components.append(component)
        regions = []
        for idx, ids in enumerate(components, 1):
            members = [cells[cid] for cid in ids]
            lats = [c["latitude"] for c in members]; lons = [c["longitude"] for c in members]
            rains = [c["precipitation_mm"] for c in members if c["precipitation_mm"] is not None]
            anomalies = [c["anomaly_pct"] for c in members if c["anomaly_pct"] is not None]
            zscores = [c["extreme_anomaly_score_z"] for c in members if c["extreme_anomaly_score_z"] is not None]
            mean_anomaly = sum(anomalies) / len(anomalies) if anomalies else None
            severity = "high anomaly" if mean_anomaly is not None and mean_anomaly >= 100 else "elevated anomaly"
            centroid_lat, centroid_lon = sum(lats)/len(lats), sum(lons)/len(lons)
            half = GRID_SPACING / 2
            polys = [[[c["longitude"]-half,c["latitude"]-half],[c["longitude"]+half,c["latitude"]-half],
                      [c["longitude"]+half,c["latitude"]+half],[c["longitude"]-half,c["latitude"]+half],
                      [c["longitude"]-half,c["latitude"]-half]] for c in members]
            area = sum((111.32 * GRID_SPACING) * (111.32 * GRID_SPACING * math.cos(math.radians(c["latitude"]))) for c in members)
            regions.append({"forecast_day": day["day"], "forecast_date": day["forecast_date"],
                            "region_id": f"D{day['day']}-R{idx:03d}", "track_id": None,
                            "centroid_lat": round(centroid_lat, 3), "centroid_lon": round(centroid_lon, 3),
                            "affected_grid_points": len(members), "cell_ids": ids,
                            "area_km2_approx": round(area), "peak_value_mm": round(max(rains), 2) if rains else None,
                            "mean_value_mm": round(sum(rains)/len(rains), 2) if rains else None,
                            "mean_anomaly_pct": round(mean_anomaly, 1) if mean_anomaly is not None else None,
                            "anomaly_score_z_mean": round(sum(zscores)/len(zscores), 2) if zscores else None,
                            "bounding_box": {"min_lat": min(lats)-half, "max_lat": max(lats)+half,
                                             "min_lon": min(lons)-half, "max_lon": max(lons)+half},
                            "severity": severity,
                            "geometry": {"type": "MultiPolygon", "coordinates": [[poly] for poly in polys]}})
        regions_by_day.append({"day": day["day"], "forecast_date": day["forecast_date"], "regions": regions})

    evolution = _build_evolution(regions_by_day)
    for day in regions_by_day:
        day["evolution"] = [e for e in evolution if e["forecast_day"] == day["day"]]
    has_missing_values = any(not c["data_valid"] for d in all_days for c in d["cells"])
    coverage_status = "partial" if errors or has_missing_values else "complete"
    baseline_freshness = "mixed" if len(baseline_origins) > 1 else next(iter(baseline_origins), "unavailable")
    baseline_models = {c.get("baseline_model", "unavailable") for d in all_days for c in d["cells"]
                       if c.get("baseline_mean_mm") is not None}
    if not baseline_models:
        baseline_model_status = "unavailable"
    elif baseline_models == {"era5"}:
        baseline_model_status = "verified ERA5"
    elif baseline_models == {"unspecified"}:
        baseline_model_status = "unverified; model unspecified in cached values"
    else:
        baseline_model_status = "mixed; includes cached values with unspecified model"
    return {"created_at": datetime.now(timezone.utc).isoformat(), "data_status": coverage_status,
            "coverage_status": coverage_status, "freshness_status": "live",
            "baseline_freshness": baseline_freshness, "cache_age_seconds": 0,
            "baseline_model_status": baseline_model_status,
            "forecast_start_time": min(forecast_timestamps) if forecast_timestamps else None,
            "forecast_end_time": max(forecast_timestamps) if forecast_timestamps else None,
            "forecast_retrieved_at": datetime.now(timezone.utc).isoformat(),
            "errors": errors[:20], "forecast_source": "ECMWF IFS via Open-Meteo ECMWF API",
            "baseline_source": ("Open-Meteo Historical Weather API (ERA5 model)"
                                if baseline_model_status == "verified ERA5" else BASELINE_SOURCE_NAME),
            "baseline_period": BASELINE_PERIOD_NAME,
            "grid": {"min_lat": GRID_MIN_LAT, "max_lat": GRID_MAX_LAT, "min_lon": GRID_MIN_LON,
                     "max_lon": GRID_MAX_LON, "spacing_degrees": GRID_SPACING,
                     "rows": len({c['row'] for c in SPATIAL_GRID}), "columns": len({c['col'] for c in SPATIAL_GRID}),
                     "configured_points": len(SPATIAL_GRID),
                     "valid_forecast_points": sum(c["precipitation_mm"] is not None for c in all_days[0]["cells"]) if all_days else 0,
                     "valid_baseline_points": sum(c["baseline_mean_mm"] is not None for c in all_days[0]["cells"]) if all_days else 0,
                     "valid_analysis_points": sum(c["data_valid"] for c in all_days[0]["cells"]) if all_days else 0},
            "variables": ["daily precipitation sum (mm)", "daily maximum 10 m wind (km/h)", "daily mean 2 m temperature (°C)"],
            "unavailable_variables": ["precipitation probability (not offered by ECMWF endpoint)"],
            "method": {"anomaly_pct": "((forecast_mm - historical seasonal mean_mm) / seasonal mean_mm) * 100; near-zero baseline uses existing 0.1 mm guard",
                       "score": "(forecast_mm - historical seasonal mean_mm) / historical seasonal standard deviation; unavailable when std < 0.1 mm",
                       "mask": f"anomaly_pct >= {ANOMALY_THRESHOLD_PCT:g} AND forecast precipitation >= {MIN_EXTREME_RAIN_MM:g} mm",
                       "connectivity": "4-neighbor connected components; components under minimum cell count excluded",
                       "min_region_cells": MIN_REGION_CELLS},
            "days": all_days, "regions_by_day": regions_by_day, "evolution": evolution}


def _haversine_km(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    r = 6371.0
    p1, p2 = math.radians(a["centroid_lat"]), math.radians(b["centroid_lat"])
    dp = math.radians(b["centroid_lat"]-a["centroid_lat"])
    dl = math.radians(b["centroid_lon"]-a["centroid_lon"])
    x = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*r*math.asin(min(1, math.sqrt(x)))


def _build_evolution(days: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    evolution = []
    next_track = 1
    previous = []
    for day in days:
        current = day["regions"]
        candidates = []
        for old in previous:
            for new in current:
                old_ids, new_ids = set(old["cell_ids"]), set(new["cell_ids"])
                overlap = len(old_ids & new_ids)
                distance = _haversine_km(old, new)
                union = len(old_ids | new_ids)
                jaccard = overlap / union if union else 0
                if overlap or distance <= GRID_SPACING * 111.32 * 1.6:
                    candidates.append((jaccard + (0.25 if distance < GRID_SPACING*111.32 else 0), old, new, distance, jaccard))
        matches, used_old, used_new = [], set(), set()
        for _, old, new, distance, jaccard in sorted(candidates, key=lambda x: x[0], reverse=True):
            if old["region_id"] in used_old or new["region_id"] in used_new:
                continue
            used_old.add(old["region_id"]); used_new.add(new["region_id"])
            new["track_id"] = old.get("track_id") or f"T{next_track:03d}"
            if not old.get("track_id"):
                next_track += 1
            delta = new["affected_grid_points"] - old["affected_grid_points"]
            events = ["persisted"]
            if distance >= 20: events.append("moved")
            if delta > 0: events.append("expanded")
            elif delta < 0: events.append("contracted")
            matches.append({"forecast_day": day["day"], "forecast_date": day["forecast_date"],
                            "track_id": new["track_id"], "from_region_id": old["region_id"],
                            "region_id": new["region_id"], "event": events,
                            "centroid_distance_km": round(distance, 1), "cell_overlap_jaccard": round(jaccard, 3),
                            "grid_point_change": delta})
        for region in current:
            if region["region_id"] not in used_new:
                region["track_id"] = f"T{next_track:03d}"; next_track += 1
                matches.append({"forecast_day": day["day"], "forecast_date": day["forecast_date"],
                                "track_id": region["track_id"], "from_region_id": None,
                                "region_id": region["region_id"], "event": ["appeared"],
                                "centroid_distance_km": None, "cell_overlap_jaccard": None, "grid_point_change": region["affected_grid_points"]})
        for old in previous:
            if old["region_id"] not in used_old:
                matches.append({"forecast_day": day["day"], "forecast_date": day["forecast_date"],
                                "track_id": old.get("track_id"), "from_region_id": old["region_id"],
                                "region_id": None, "event": ["disappeared"],
                                "centroid_distance_km": None, "cell_overlap_jaccard": None, "grid_point_change": -old["affected_grid_points"]})
        evolution.extend(matches)
        previous = current
    return evolution


def _spatial_day(day: int) -> Dict[str, Any]:
    analysis = _fetch_spatial_inputs()
    if day < 1 or day > len(analysis["days"]):
        raise HTTPException(status_code=404, detail=f"Forecast day must be from 1 to {len(analysis['days'])}.")
    return analysis, analysis["days"][day - 1], analysis["regions_by_day"][day - 1]


def _spatial_compat_anomaly(location: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt actual ECMWF/ERA5 grid data to the legacy point response shape."""
    analysis = _fetch_spatial_inputs()
    first_day = analysis["days"][0]
    valid = [c for c in first_day["cells"] if c.get("data_valid")]
    if not valid:
        raise HTTPException(status_code=503, detail="Point APIs and the cached ECMWF/ERA5 spatial grid are unavailable.")
    nearest = min(valid, key=lambda c: (c["latitude"]-location["latitude"])**2 + (c["longitude"]-location["longitude"])**2)
    series = [next(c for c in d["cells"] if c["id"] == nearest["id"]) for d in analysis["days"]]
    first = series[0]
    rain_series = [c.get("precipitation_mm") for c in series]
    valid_rain = [v for v in rain_series if v is not None]
    wind_series = [c.get("wind_speed_max_kmh") for c in series]
    valid_wind = [v for v in wind_series if v is not None]
    normal, rain = first.get("baseline_mean_mm"), first.get("precipitation_mm")
    anomaly_pct = calculate_anomaly(rain, normal) if rain is not None and normal is not None else 0
    return {
        "id": location["id"], "region": location["region"], "severity": classify_severity(anomaly_pct),
        "normal_mm": normal, "predicted_mm": rain, "anomaly_pct": anomaly_pct,
        "latitude": first["latitude"], "longitude": first["longitude"],
        "data_source": analysis["forecast_source"], "fetched_at": analysis["created_at"],
        "forecast_10day_mm": round(sum(valid_rain), 1) if len(valid_rain) == len(series) else None,
        "baseline_source": analysis["baseline_source"], "baseline_model_status": analysis.get("baseline_model_status", "unverified; model unspecified"),
        "baseline_period": analysis["baseline_period"],
        "baseline_method": BASELINE_METHOD_NAME, "baseline_sample_days": first.get("baseline_sample_days", 0),
        "anomaly_method": "percentage rainfall anomaly ((forecast - historical seasonal mean) / mean) * 100 (NOT EFI)",
        "forecast_timestamps": [], "hourly_precipitation": [],
        "hourly_precipitation_probability": [], "hourly_wind_speed": [],
        "daily_timestamps": [d["forecast_date"] for d in analysis["days"]],
        "daily_precipitation_sum": rain_series,
        "max_hourly_rain_mm": None, "max_wind_kmh": max(valid_wind) if valid_wind else None,
        "max_precip_prob_pct": None, "spatial_grid_fallback": True,
        "spatial_grid_id": first["id"], "spatial_grid_status": analysis["data_status"],
    }


def _spatial_compat_tracking(anomaly: Dict[str, Any]) -> Dict[str, Any]:
    analysis = _fetch_spatial_inputs()
    grid_id = anomaly["spatial_grid_id"]
    days_list = []
    for index, d in enumerate(analysis["days"]):
        cell = next(c for c in d["cells"] if c["id"] == grid_id)
        rain, normal = cell.get("precipitation_mm"), cell.get("baseline_mean_mm")
        anomaly_pct = calculate_anomaly(rain, normal) if rain is not None and normal is not None else None
        days_list.append({"day": index + 1, "forecast_date": d["forecast_date"], "rainfall_mm": rain,
                          "baseline_mm": normal, "anomaly_pct": anomaly_pct,
                          "severity": classify_severity(anomaly_pct) if anomaly_pct is not None else "unavailable",
                          "precip_probability_pct": None, "wind_speed_kmh": cell.get("wind_speed_max_kmh"),
                          "latitude": cell["latitude"], "longitude": cell["longitude"],
                          "screening_radius_km": None,
                          "footprint_status": "Use connected-cell GeoJSON at /api/threat-footprint; no point circle is represented as a boundary",
                          "spatial_status": "nearest valid ECMWF/ERA5 grid cell"})
    valid = [d for d in days_list if d["anomaly_pct"] is not None]
    peak = max(valid, key=lambda d: d["anomaly_pct"]) if valid else None
    anomalous = [d["day"] for d in valid if d["anomaly_pct"] >= 25]
    return {"region": anomaly["region"], "tracking_mode": "REAL ECMWF GRID-POINT TIME SERIES",
            "forecast_source": analysis["forecast_source"], "baseline_source": analysis["baseline_source"],
            "spatial_tracking_status": "Grid-point series; connected-region matching is available from /api/threat-evolution",
            "days": days_list,
            "event_summary": {"peak_anomaly_pct": peak["anomaly_pct"] if peak else None,
                              "peak_day": peak["day"] if peak else None,
                              "peak_forecast_date": peak["forecast_date"] if peak else "",
                              "persistence_days": len(anomalous),
                              "first_anomalous_day": anomalous[0] if anomalous else None,
                              "last_anomalous_day": anomalous[-1] if anomalous else None,
                              "max_rainfall_mm": max((d["rainfall_mm"] for d in days_list if d["rainfall_mm"] is not None), default=None),
                              "thresholds_note": "Illustrative prototype anomaly bands, not official warning criteria."}}


@app.get("/api/spatial-field")
def spatial_field(day: int = Query(1, ge=1, le=10)):
    analysis, day_data, regions = _spatial_day(day)
    features = []
    half = GRID_SPACING / 2
    for c in day_data["cells"]:
        if not c["data_valid"]:
            continue
        coords = [[[c["longitude"]-half,c["latitude"]-half],[c["longitude"]+half,c["latitude"]-half],
                   [c["longitude"]+half,c["latitude"]+half],[c["longitude"]-half,c["latitude"]+half],
                   [c["longitude"]-half,c["latitude"]-half]]]
        features.append({"type": "Feature", "id": c["id"], "geometry": {"type": "Polygon", "coordinates": coords},
                         "properties": {k: c[k] for k in ("id", "latitude", "longitude", "forecast_date", "precipitation_mm",
            "baseline_mean_mm", "baseline_std_mm", "baseline_sample_days", "anomaly_pct", "extreme_anomaly_score_z",
                             "wind_speed_max_kmh", "temperature_mean_c", "anomaly_mask")}})
    return {"day": day, "forecast_date": day_data["forecast_date"], "grid": analysis["grid"],
            "data_status": analysis["data_status"], "stale_age_minutes": analysis.get("stale_age_minutes"), "errors": analysis["errors"],
            "coverage_status": analysis.get("coverage_status", analysis["data_status"]),
            "freshness_status": analysis.get("freshness_status", "cached"),
            "baseline_freshness": analysis.get("baseline_freshness", "unavailable"),
            "baseline_model_status": analysis.get("baseline_model_status", "unverified; model unspecified"),
            "cache_age_seconds": analysis.get("cache_age_seconds"),
            "forecast_retrieved_at": analysis.get("forecast_retrieved_at"),
            "forecast_start_time": analysis.get("forecast_start_time"),
            "forecast_end_time": analysis.get("forecast_end_time"),
            "forecast_source": analysis["forecast_source"], "baseline_source": analysis["baseline_source"],
            "baseline_period": analysis["baseline_period"], "variables": analysis["variables"],
            "unavailable_variables": analysis["unavailable_variables"], "method": analysis["method"],
            "valid_cells": len(features), "features": features,
            "regions": regions["regions"], "evolution": regions["evolution"]}


@app.get("/api/threat-regions")
def threat_regions(day: int = Query(1, ge=1, le=10)):
    analysis, day_data, region_data = _spatial_day(day)
    return {"day": day, "forecast_date": day_data["forecast_date"], "data_status": analysis["data_status"],
            "freshness_status": analysis.get("freshness_status", "cached"),
            "coverage_status": analysis.get("coverage_status", analysis["data_status"]),
            "baseline_model_status": analysis.get("baseline_model_status", "unverified; model unspecified"),
            "region_count": len(region_data["regions"]), "regions": region_data["regions"],
            "evolution": region_data["evolution"], "thresholds": analysis["method"]}


@app.get("/api/threat-evolution")
def threat_evolution():
    analysis = _fetch_spatial_inputs()
    return {"forecast_source": analysis["forecast_source"], "data_status": analysis["data_status"],
            "freshness_status": analysis.get("freshness_status", "cached"),
            "coverage_status": analysis.get("coverage_status", analysis["data_status"]),
            "baseline_model_status": analysis.get("baseline_model_status", "unverified; model unspecified"),
            "days": [{"day": d["day"], "forecast_date": d["forecast_date"], "regions": d["regions"]} for d in analysis["regions_by_day"]],
            "events": analysis["evolution"]}


@app.get("/api/threat-footprint")
def threat_footprint(day: int = Query(1, ge=1, le=10), region_id: Optional[str] = None):
    analysis, day_data, region_data = _spatial_day(day)
    regions = [r for r in region_data["regions"] if region_id is None or r["region_id"] == region_id]
    features = [{"type": "Feature", "id": r["region_id"], "geometry": r["geometry"],
                 "properties": {k: v for k, v in r.items() if k != "geometry"}} for r in regions]
    return {"type": "FeatureCollection", "day": day, "forecast_date": day_data["forecast_date"],
            "data_status": analysis["data_status"],
            "freshness_status": analysis.get("freshness_status", "cached"),
            "coverage_status": analysis.get("coverage_status", analysis["data_status"]), "features": features}


@app.get("/api/health")
def health():
    """
    System health, data sources verification, and cache status.
    """
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "data_source": "ECMWF IFS via Open-Meteo ECMWF API (spatial); Open-Meteo Forecast API (legacy point endpoints)",
        "baseline_source": BASELINE_SOURCE_NAME,
        "baseline_period": BASELINE_PERIOD_NAME,
        "baseline_method": BASELINE_METHOD_NAME,
        "cache": {
            "cached_forecast_locations": list(_FORECAST_CACHE.keys()),
            "cached_baseline_keys": list(_BASELINE_CACHE.keys()),
            "ttl_seconds": CACHE_TTL_SECONDS,
        },
        "spatial_data": {
            "status": (_SPATIAL_CACHE.get("analysis", {}).get("value", {}).get("data_status")
                       if _SPATIAL_CACHE.get("analysis") else "not_loaded"),
            "freshness_status": (_SPATIAL_CACHE.get("analysis", {}).get("last_delivery_status",
                                  _SPATIAL_CACHE.get("analysis", {}).get("value", {}).get("freshness_status"))
                                 if _SPATIAL_CACHE.get("analysis") else "unknown"),
            "cache_ttl_seconds": GRID_CACHE_TTL,
            "forecast_retrieved_at": (_SPATIAL_CACHE.get("analysis", {}).get("value", {}).get("forecast_retrieved_at")
                                      if _SPATIAL_CACHE.get("analysis") else None),
            "forecast_start_time": (_SPATIAL_CACHE.get("analysis", {}).get("value", {}).get("forecast_start_time")
                                    if _SPATIAL_CACHE.get("analysis") else None),
            "forecast_end_time": (_SPATIAL_CACHE.get("analysis", {}).get("value", {}).get("forecast_end_time")
                                  if _SPATIAL_CACHE.get("analysis") else None),
        },
        "monitored_locations": [loc["region"] for loc in LOCATIONS],
        "pipeline_status": {
            "forecast_ingestion": "REAL (Open-Meteo ECMWF API spatial field plus existing point API)",
            "historical_baseline": "REAL (Open-Meteo Archive; new requests explicitly select ERA5; legacy cached values are model-unverified)",
            "rainfall_anomaly": "REAL / DATA-BASED (Percentage anomaly, explicitly NOT EFI)",
            "spatial_tracking": "REAL ECMWF grid anomaly components and consecutive-day region matching (prototype method)",
            "screening_footprint": "GeoJSON cells grouped from thresholded spatial anomaly mask; not a storm-system boundary",
            "localization_downscaling": "ECMWF field sampled on configured regional grid; no 5 km downscaling claimed",
            "physics_checks": "BASIC range/aggregation consistency only",
            "ensemble_uncertainty": "UNAVAILABLE (deterministic ECMWF feed; no ensemble members connected)",
        },
    }


@app.get("/api/forecast", response_model=List[AnomalyOut])
def forecast():
    """
    Get 10-day real forecast ingestion and ERA5 seasonal baseline anomalies for all locations.
    """
    return get_all_anomalies()


@app.get("/api/anomaly/{anomaly_id}", response_model=AnomalyOut)
def anomaly_detail(anomaly_id: str):
    """
    Get detailed real forecast metrics and ERA5 anomaly classification for one location.
    """
    return _find_anomaly(anomaly_id)


@app.get("/api/threat-zones")
def threat_zones():
    """
    Current real forecast-derived threat zones for map rendering.
    """
    return get_all_anomalies()


@app.get("/api/tracking/{anomaly_id}")
def tracking(anomaly_id: str):
    """
    Daily forecast time-series and historical comparison for one fixed point.
    No spatial storm track is inferred.
    """
    return get_tracking_data(anomaly_id)


@app.get("/api/downscale/{anomaly_id}")
def downscale(anomaly_id: str):
    """
    Localization status; unavailable until gridded forecast input is connected.
    """
    return downscale_status(anomaly_id)


@app.get("/api/uncertainty/{anomaly_id}")
def uncertainty(anomaly_id: str):
    """
    Ensemble uncertainty status; unavailable until ensemble members are connected.
    """
    return uncertainty_status(anomaly_id)


@app.get("/api/physics/{anomaly_id}")
def physics(anomaly_id: str):
    """Run basic input-range and time-aggregation checks on the live forecast."""
    return check_physical_consistency(_find_anomaly(anomaly_id))


@app.get("/api/alerts")
def alerts(severity: Optional[str] = None):
    """
    Forecast-derived alert candidates; thresholds are prototype screening rules.
    """
    anomalies = get_all_anomalies()
    output = []

    for a in anomalies:
        # Emit only elevated positive rainfall-anomaly candidates.
        if a["anomaly_pct"] < 25:
            continue
        if severity and isinstance(severity, str) and a["severity"] != severity.lower():
            continue

        output.append({
            "region": a["region"],
            "severity": a["severity"].upper(),
            "expected_rainfall_mm": a["predicted_mm"],
            "normal_rainfall_mm": a["normal_mm"],
            "anomaly_pct": a["anomaly_pct"],
            "valid_until": a.get("daily_timestamps", [None])[0],
            "latitude": a["latitude"],
            "longitude": a["longitude"],
            "issued_at": datetime.now(timezone.utc).isoformat(),
            "fetched_at": a.get("fetched_at"),
            "data_source": DATA_SOURCE_NAME,
            "baseline_source": BASELINE_SOURCE_NAME,
            "baseline_period": BASELINE_PERIOD_NAME,
            "baseline_method": BASELINE_METHOD_NAME,
            "alert_status": "prototype screening candidate; not an official warning",
        })

    return output


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
