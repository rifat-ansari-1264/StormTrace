"""
StormTrace live check
=====================
Calls the locally running backend (http://127.0.0.1:8000) and reports, per endpoint,
PASS / FAIL plus the data provenance (forecast source, baseline source), anomaly,
threat severity / score and uncertainty status.

Usage (backend must already be running):
    python live_check.py
    python live_check.py --base http://127.0.0.1:8000 --id odisha-coast

Only Python's standard library is used. Nothing is invented: any field the backend
does not return is printed as "not provided".
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

NA = "not provided"


def get(base: str, path: str, timeout: int = 180) -> Tuple[Optional[int], Any, float, str]:
    """Returns (status, parsed_json_or_None, seconds, error_text)."""
    t0 = time.time()
    try:
        with urllib.request.urlopen(base + path, timeout=timeout) as r:
            body = r.read().decode("utf-8")
            try:
                return r.status, json.loads(body), time.time() - t0, ""
            except ValueError:
                return r.status, None, time.time() - t0, "response is not valid JSON"
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        return e.code, None, time.time() - t0, detail
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return None, None, time.time() - t0, f"cannot connect: {e}"


def pick(d: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if isinstance(d, dict) and d.get(k) not in (None, ""):
            return d[k]
    return NA


def need(d: Any, keys: List[str]) -> List[str]:
    return [k for k in keys if not isinstance(d, dict) or k not in d]


# ---------------------------------------------------------------- per-endpoint checks
# Each returns (problems, lines_to_print). Empty problems list == PASS.

def chk_health(d):
    p = [] if isinstance(d, dict) and d.get("status") == "ok" else ["status != ok"]
    ps = d.get("model_status") or d.get("pipeline_status") or {}
    lines = [f"forecast source : {pick(d, 'forecast_source', 'data_source')}",
             f"baseline source : {pick(d, 'baseline_source')}  [{pick(d, 'baseline_period')}]"]
    lines += [f"  {k}: {v}" for k, v in ps.items()]
    return p, lines


def zone_line(z: Dict[str, Any]) -> str:
    return (f"{str(pick(z, 'region')):<22} severity={str(pick(z, 'severity')):<9} "
            f"score={str(pick(z, 'threat_score')):<6} predicted={pick(z, 'predicted_mm')} mm  "
            f"normal={pick(z, 'normal_mm')} mm  anomaly={pick(z, 'anomaly_pct')}%")


def chk_zone_list(d):
    if not isinstance(d, list) or not d:
        return ["expected a non-empty list"], []
    p = []
    for z in d:
        miss = need(z, ["id", "region", "severity", "anomaly_pct"])
        if miss:
            p.append(f"{z.get('id', '?')}: missing {miss}")
    src = d[0]
    lines = [f"forecast source : {pick(src, 'forecast_source', 'data_source')}",
             f"baseline source : {pick(src, 'baseline_source')}  [{pick(src, 'baseline_period')}]",
             f"fetched_at      : {pick(src, 'fetched_at')}"]
    lines += ["  " + zone_line(z) for z in d]
    return p, lines


def chk_anomaly(d):
    p = [f"missing {m}" for m in need(d, ["region", "normal_mm", "predicted_mm", "anomaly_pct", "severity"])]
    lines = [f"region          : {pick(d, 'region')}",
             f"forecast source : {pick(d, 'forecast_source', 'data_source')}",
             f"baseline source : {pick(d, 'baseline_source')}  [{pick(d, 'baseline_period')}]",
             f"anomaly         : {pick(d, 'anomaly_pct')}%  (forecast {pick(d, 'predicted_mm')} mm vs normal {pick(d, 'normal_mm')} mm)",
             f"anomaly method  : {pick(d, 'anomaly_method')}",
             f"severity        : {pick(d, 'severity')}    threat score: {pick(d, 'threat_score')}"]
    return p, lines


def chk_tracking(d):
    days = d.get("days") if isinstance(d, dict) else None
    if not isinstance(days, list) or not days:
        return ["no 'days' list"], []
    p = [f"day {x.get('day', '?')} missing {m}" for x in days
         for m in need(x, ["rainfall_mm", "baseline_mm", "anomaly_pct"])][:5]
    s = d.get("event_summary", {})
    lines = [f"forecast source : {pick(d, 'forecast_source')}",
             f"days returned   : {len(days)}",
             f"peak anomaly    : {pick(s, 'peak_anomaly_pct')}% on day {pick(s, 'peak_day')} ({pick(s, 'peak_forecast_date', 'peak_date')})",
             f"persistence     : {pick(s, 'persistence_days')} day(s)",
             f"max rainfall    : {pick(s, 'max_rainfall_mm')} mm on day {pick(s, 'max_rainfall_day')}"]
    lines += [f"  D{x['day']:<2} {x.get('forecast_date', '')}  rain={x.get('rainfall_mm')} mm  "
              f"normal={x.get('baseline_mm')} mm  anomaly={x.get('anomaly_pct')}%  {x.get('severity', '')}"
              for x in days]
    traj = d.get("simulated_trajectory", {}).get("status") or d.get("model_status")
    lines.append(f"spatial track   : {traj if traj else NA}")
    return p, lines


def chk_downscale(d):
    status = pick(d, "method_status", "model_status")
    if isinstance(d, dict) and d.get("status") == "unavailable":
        return [], ["localization     : honestly unavailable", f"reason          : {pick(d, 'reason')}"]
    p = [f"missing {m}" for m in need(d, ["coarse", "fine", "method_status"])]
    return p, [f"coarse          : {pick(d.get('coarse', {}), 'resolution_km')} km grid",
               f"fine            : {pick(d.get('fine', {}), 'resolution_km')} km grid",
               f"method status   : {status}"]


def chk_physics(d):
    checks = d.get("checks") if isinstance(d, dict) else None
    if not isinstance(checks, list) or not checks:
        return ["no physical consistency checks returned"], []
    lines = [f"scope           : {pick(d, 'scope')}"]
    lines += [f"  {c.get('status', '?')}: {c.get('name', 'unnamed check')}" for c in checks]
    return [], lines


def chk_uncertainty(d):
    if not isinstance(d, dict):
        return ["not a JSON object"], []
    lines = []
    if d.get("ensemble_status") == "ok":
        fs = d.get("focus_stats", {})
        lines += ["ENSEMBLE-BASED (real)",
                  f"source          : {pick(d, 'uncertainty_source')}",
                  f"members         : {pick(d, 'members')}",
                  f"confidence      : {pick(d, 'confidence_pct')}%  ({pick(d, 'confidence_definition')})",
                  f"mean / std      : {pick(fs, 'mean_mm')} / {pick(fs, 'std_mm')} mm",
                  f"p10 / p50 / p90 : {pick(fs, 'p10_mm')} / {pick(fs, 'p50_mm')} / {pick(fs, 'p90_mm')} mm",
                  f"P(>=15.6 mm)    : {pick(fs, 'prob_ge_15_6mm_pct')}%"]
        return [], lines
    if d.get("ensemble_status") == "unavailable":
        lines += ["ENSEMBLE UNAVAILABLE (backend reports this explicitly)",
                  f"reason          : {pick(d, 'reason')}"]
        return [], lines
    src = str(pick(d, "source", "uncertainty_source"))
    mod = str(pick(d, "module_status"))
    lines += ["SIMULATED uncertainty (NOT a real ensemble)" if "SIM" in (src + mod).upper() else "source not stated",
              f"source          : {src}", f"module status   : {mod}",
              f"confidence      : {pick(d, 'confidence_pct')}%"]
    return [], lines


def chk_alerts(d):
    if not isinstance(d, list):
        return ["expected a list"], []
    p = []
    for a in d:
        miss = need(a, ["region", "severity"])
        if miss:
            p.append(f"alert missing {miss}")
    lines = [f"alert count     : {len(d)}" + ("  (no elevated threats right now - valid outcome)" if not d else "")]
    for a in d:
        rain = pick(a, "forecast_rainfall_mm", "expected_rainfall_mm")
        lines.append(f"  {str(a.get('severity')):<9} {str(a.get('region')):<22} rain={rain} mm  "
                     f"anomaly={pick(a, 'anomaly_pct')}%  score={pick(a, 'threat_score')}  "
                     f"peak={pick(a, 'peak_date')}  confidence={pick(a, 'confidence_pct')}")
    if d:
        lines += [f"forecast source : {pick(d[0], 'forecast_source', 'data_source')}",
                  f"baseline source : {pick(d[0], 'baseline_source')}"]
    return p, lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--id", default="odisha-coast")
    a = ap.parse_args()
    i = a.id
    checks: List[Tuple[str, Callable]] = [
        ("/api/health", chk_health), ("/api/forecast", chk_zone_list), (f"/api/anomaly/{i}", chk_anomaly),
        ("/api/threat-zones", chk_zone_list), (f"/api/tracking/{i}", chk_tracking),
        (f"/api/downscale/{i}", chk_downscale), (f"/api/physics/{i}", chk_physics),
        (f"/api/uncertainty/{i}", chk_uncertainty),
        ("/api/alerts", chk_alerts),
    ]
    results = []
    for n, (path, fn) in enumerate(checks, 1):
        print("=" * 78)
        print(f"{n}. GET {path}")
        status, data, secs, err = get(a.base, path)
        if status != 200 or data is None:
            print(f"   FAIL  HTTP {status}  ({secs:.1f}s)  {err}")
            results.append((path, False))
            continue
        problems, lines = fn(data)
        for ln in lines:
            print("   " + ln)
        ok = not problems
        print(f"   {'PASS' if ok else 'FAIL'}  HTTP 200  ({secs:.1f}s)" + ("" if ok else "  -> " + "; ".join(problems)))
        results.append((path, ok))

    print("=" * 78)
    t0 = time.time()
    get(a.base, "/api/threat-zones")
    print(f"Cache check: repeat /api/threat-zones took {time.time() - t0:.2f}s (should be near-instant)")
    print("SUMMARY")
    for path, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {path}")
    failed = [p for p, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} endpoints passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
