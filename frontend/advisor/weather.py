"""Weather for a field from Open-Meteo (free, no key): growing degree days, water balance, heat and wet spells."""
from __future__ import annotations

import hashlib
import io
import json
from datetime import date, timedelta
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
    key = hashlib.md5(f"{lat:.2f},{lon:.2f},{start},{end},{date.today()}".encode()).hexdigest()
    path = config.CACHE_DIR / f"wx_{key}.json"
    if path.exists():
        df = pd.read_json(io.StringIO(path.read_text(encoding="utf-8")), orient="split")
        df["date"] = pd.to_datetime(df["date"])
        return df

    base = {"latitude": round(lat, 4), "longitude": round(lon, 4), "daily": DAILY, "timezone": "auto"}
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


def get_weather(lat: float, lon: float, sowing: date, stage: str, irrigated: bool) -> Optional[dict]:
    """Weather block for the evidence packet, or None when the service cannot be reached."""
    try:
        df = fetch_daily(lat, lon, sowing, date.today())
        if df.empty:
            return None
        return {"summary": summarize(df, stage, irrigated), "weekly": weekly_series(df), "source": "Open-Meteo"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200]}
