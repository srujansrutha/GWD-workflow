"""Weather for a field from Open-Meteo (free, no key): growing degree days, water balance, heat and wet spells."""
from __future__ import annotations

import hashlib
import io
import json
import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests

from . import config

DAILY = "temperature_2m_max,temperature_2m_min,precipitation_sum,et0_fao_evapotranspiration"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_LAG_DAYS = 8       # the archive trails real time; the forecast endpoint covers the most recent days


def _get(url: str, params: dict) -> pd.DataFrame:
    r = requests.get(url, params=params, headers={"User-Agent": config.USER_AGENT}, timeout=config.HTTP_TIMEOUT)
    r.raise_for_status()
    d = r.json().get("daily", {})
    df = pd.DataFrame({"date": pd.to_datetime(d["time"]), "tmax": d["temperature_2m_max"],
                       "tmin": d["temperature_2m_min"], "rain": d["precipitation_sum"],
                       "et0": d["et0_fao_evapotranspiration"]})
    return df


def fetch_daily(lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
    """Daily weather between two dates (inclusive), merged from the archive and the recent forecast feed."""
    key = hashlib.md5(f"{lat:.2f},{lon:.2f},{start},{end},{date.today()}".encode(), usedforsecurity=False).hexdigest()
    path = config.CACHE_DIR / f"wx_{key}.json"
    if path.exists():
        df = pd.read_json(io.StringIO(path.read_text(encoding="utf-8")), orient="split")
        df["date"] = pd.to_datetime(df["date"])
        return df

    base = {"latitude": round(lat, 2), "longitude": round(lon, 2), "daily": DAILY, "timezone": "auto"}   # weather grids are 9 km or coarser
    parts = []
    split = date.today() - timedelta(days=ARCHIVE_LAG_DAYS)
    if start <= split:
        parts.append(_get(ARCHIVE_URL, {**base, "start_date": start.isoformat(),
                                        "end_date": min(end, split).isoformat()}))
    if end > split:
        past = min(92, (date.today() - max(start, split + timedelta(days=1))).days + 1)
        parts.append(_get(FORECAST_URL, {**base, "past_days": max(past, 1), "forecast_days": 1}))
    df = (pd.concat(parts).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True))
    df = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)].reset_index(drop=True)
    path.write_text(df.to_json(orient="split", date_format="iso"), encoding="utf-8")
    return df


def summarize(df: pd.DataFrame, stage: str, irrigated: bool) -> dict:
    """Numbers the agronomy engine uses. Missing days are ignored, and ``days`` says how many were real."""
    d = df.dropna(subset=["tmax", "tmin"]).copy()
    d["tmean"] = (d["tmax"] + d["tmin"]) / 2
    d["gdd"] = d["tmean"].clip(lower=0)                          # base 0 C, a common simple choice for wheat
    last28, last21, last14 = d.tail(28), d.tail(21), d.tail(14)
    rain28 = float(last28["rain"].fillna(0).sum())
    et28 = float(last28["et0"].fillna(0).sum())
    out = {
        "days": int(len(d)),
        "gdd_since_sowing": round(float(d["gdd"].sum())),
        "rain_since_sowing_mm": round(float(d["rain"].fillna(0).sum())),
        "rain_28d_mm": round(rain28),
        "et0_28d_mm": round(et28),
        "water_balance_28d_mm": round(rain28 - et28),            # rain minus reference crop water use
        "heat_days_21d": int((last21["tmax"] >= 30).sum()),
        "rainy_days_14d": int((last14["rain"].fillna(0) >= 1).sum()),
        "tmax_peak_21d": round(float(last21["tmax"].max()), 1) if len(last21) else None,
        "last_date": d["date"].max().date().isoformat() if len(d) else None,
    }
    return out


def weekly_series(df: pd.DataFrame) -> list[dict]:
    """Weekly rain, reference water use and mean temperature, for the chart."""
    d = df.copy().set_index("date")
    w = d.resample("W").agg({"rain": "sum", "et0": "sum", "tmax": "max", "tmin": "min"}).dropna(how="all")
    return [{"week": i.date().isoformat(), "rain_mm": round(float(r.rain), 1), "et0_mm": round(float(r.et0), 1),
             "tmax": round(float(r.tmax), 1), "tmin": round(float(r.tmin), 1)} for i, r in w.iterrows()]


# ----------------------------------------------------------------------------- forecast for the next days
FORECAST_DAILY = DAILY + ",precipitation_probability_max,relative_humidity_2m_mean"


def _num(x) -> Optional[float]:
    """A number, or None for a missing value (None, NaN, text)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def day_phrase(day: date, today: date) -> str:
    """'tomorrow' or 'on Friday'. Only used for days 1 to 5, where weekday names cannot repeat."""
    return "tomorrow" if (day - today).days == 1 else f"on {day.strftime('%A')}"


def fetch_forecast(lat: float, lon: float) -> dict:
    """Daily forecast for tomorrow and the days after, from Open-Meteo. Kept for a few hours, then fetched again."""
    bucket = int(time.time() // (3600 * config.FORECAST_CACHE_HOURS))
    key = hashlib.md5(f"fc,{lat:.2f},{lon:.2f},{bucket}".encode(), usedforsecurity=False).hexdigest()
    path = config.CACHE_DIR / f"fc_{key}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    r = requests.get(FORECAST_URL, params={"latitude": round(lat, 2), "longitude": round(lon, 2), "daily": FORECAST_DAILY,
                                           "timezone": "auto", "forecast_days": config.FORECAST_DAYS + 1},
                     headers={"User-Agent": config.USER_AGENT}, timeout=config.HTTP_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    d = j["daily"]
    n = len(d["time"])
    col = {k: (d.get(k) or [None] * n) for k in ("temperature_2m_max", "temperature_2m_min", "precipitation_sum",
                                                 "precipitation_probability_max", "relative_humidity_2m_mean",
                                                 "et0_fao_evapotranspiration")}
    # "tomorrow" means tomorrow at the field, which can be another calendar day than on this computer
    local_today = (datetime.now(timezone.utc) + timedelta(seconds=int(j.get("utc_offset_seconds") or 0))).date()
    days = []
    for i, t in enumerate(d["time"]):
        if date.fromisoformat(t) <= local_today:
            continue
        days.append({"date": t, "tmax": _num(col["temperature_2m_max"][i]), "tmin": _num(col["temperature_2m_min"][i]),
                     "rain_mm": _num(col["precipitation_sum"][i]), "chance_pct": _num(col["precipitation_probability_max"][i]),
                     "humidity_pct": _num(col["relative_humidity_2m_mean"][i]), "et0_mm": _num(col["et0_fao_evapotranspiration"][i])})
    out = {"days": days[:config.FORECAST_DAYS], "local_today": local_today.isoformat(),
           "fetched": datetime.now(timezone.utc).isoformat(timespec="minutes"), "source": "Open-Meteo forecast"}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out), encoding="utf-8")
    tmp.replace(path)                                  # whole file or nothing, even with two sessions at once
    return out


def summarize_forecast(fc: dict) -> Optional[dict]:
    """Numbers the rules use, from the first few days only. None when too few days have a rain figure."""
    days = fc.get("days") or []
    act = days[:config.FORECAST_ACTION_DAYS]
    if sum(1 for d in act if d.get("rain_mm") is not None) < 3:
        return None
    today = date.fromisoformat(fc["local_today"])
    cfg = config

    def rain(d):
        return max(0.0, d.get("rain_mm") or 0.0)         # a negative figure from the service must not cancel real rain

    def chance_ok(c):
        return c is None or c >= cfg.FORECAST_MIN_CHANCE

    first3 = act[:3]
    r3, r5 = sum(rain(d) for d in first3), sum(rain(d) for d in act)
    wet3, wet5 = max(first3, key=rain), max(act, key=rain)
    hot = [d for d in act if d.get("tmax") is not None and d["tmax"] >= cfg.FORECAST_HOT_C]
    warm = [d for d in act if d.get("tmax") is not None]
    peak = max(warm, key=lambda d: d["tmax"]) if warm else None
    warm_wet = [d for d in act if rain(d) >= cfg.FORECAST_RAIN_DAY_MM and d.get("tmax") is not None
                and d["tmax"] >= cfg.FORECAST_WET_MIN_TMAX
                and (d.get("humidity_pct") is None or d["humidity_pct"] >= cfg.FORECAST_WET_MIN_RH)]
    chances = [d["chance_pct"] for d in act if d.get("chance_pct") is not None]
    missing_rain = any(d.get("rain_mm") is None for d in act)       # a missing figure must never look like "dry"
    return {
        "days_available": len(days), "action_days": len(act),
        "rain_3d_mm": round(r3, 1), "rain_5d_mm": round(r5, 1),
        "rain_days_5d": sum(1 for d in act if rain(d) >= cfg.FORECAST_RAIN_DAY_MM),
        "chance_5d_pct": round(max(chances)) if chances else None,
        "wettest": {"mm": round(rain(wet5), 1), "when": day_phrase(date.fromisoformat(wet5["date"]), today),
                    "chance_pct": None if wet5.get("chance_pct") is None else round(wet5["chance_pct"])},
        "hot_days_5d": len(hot),
        "peak_tmax_5d": round(peak["tmax"], 1) if peak else None,
        "peak_when": day_phrase(date.fromisoformat(peak["date"]), today) if peak else None,
        "warm_wet_days_5d": len(warm_wet),
        "dry_5d": (not missing_rain) and r5 < cfg.FORECAST_DRY_MM,
        "rain_soon_3d": r3 >= cfg.FORECAST_HEAVY_RAIN_MM and chance_ok(wet3.get("chance_pct")),
        "rain_sig_5d": r5 >= cfg.FORECAST_HEAVY_RAIN_MM and chance_ok(wet5.get("chance_pct")),
        "missing_days": sum(1 for d in act if d.get("rain_mm") is None or d.get("tmax") is None),
    }


def get_forecast(lat: float, lon: float) -> dict:
    """Forecast block for the evidence packet, or {"error": ...}. A failure here never stops the report."""
    try:
        fc = fetch_forecast(lat, lon)
        summary = summarize_forecast(fc)
        if summary is None:
            return {"error": "the forecast service returned fewer than 3 usable days"}
        return {**fc, "summary": summary}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def forecast_age_hours(fc: Optional[dict]) -> Optional[float]:
    """How old a saved forecast is, or None when that cannot be told."""
    try:
        return (datetime.now(timezone.utc) - datetime.fromisoformat(fc["fetched"])).total_seconds() / 3600
    except (TypeError, KeyError, ValueError):
        return None


def get_weather(lat: float, lon: float, sowing: date, stage: str, irrigated: bool) -> Optional[dict]:
    """Weather block for the evidence packet, or None when the service cannot be reached."""
    try:
        df = fetch_daily(lat, lon, sowing, date.today())
        if df.empty:
            return None
        return {"summary": summarize(df, stage, irrigated), "weekly": weekly_series(df), "source": "Open-Meteo"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200]}
