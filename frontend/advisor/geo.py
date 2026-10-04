"""Location work: GPS from photo files, distances, field area, point-in-polygon, reverse geocoding."""
from __future__ import annotations

import hashlib
import io
import json
import math
from datetime import datetime
from typing import Optional

import numpy as np
import requests
from PIL import Image

from . import config

R_EARTH = 6_371_008.8  # metres


# ----------------------------------------------------------------------------- EXIF
def _dms_to_deg(dms, ref) -> Optional[float]:
    try:
        d, m, s = (float(x) for x in dms)
        deg = d + m / 60 + s / 3600
        return -deg if str(ref).upper() in ("S", "W") else deg
    except Exception:  # noqa: BLE001
        return None


def read_exif(raw: bytes) -> dict:
    """GPS position, accuracy and capture time stored in a photo file. Empty values when missing."""
    out = {"lat": None, "lon": None, "accuracy_m": None, "time": None, "has_gps": False}
    try:
        im = Image.open(io.BytesIO(raw))
        exif = im.getexif()
        gps = exif.get_ifd(0x8825) if exif else {}
        if gps:
            lat = _dms_to_deg(gps.get(2), gps.get(1)) if gps.get(2) else None
            lon = _dms_to_deg(gps.get(4), gps.get(3)) if gps.get(4) else None
            if lat is not None and lon is not None and not (lat == 0 and lon == 0):
                out.update(lat=lat, lon=lon, has_gps=True)
            if gps.get(31) is not None:               # GPSHPositioningError, metres
                try:
                    out["accuracy_m"] = float(gps.get(31))
                except Exception:  # noqa: BLE001
                    pass
        stamp = None
        try:
            stamp = exif.get_ifd(0x8769).get(0x9003)  # DateTimeOriginal
        except Exception:  # noqa: BLE001
            pass
        stamp = stamp or exif.get(0x0132)
        if stamp:
            try:
                out["time"] = datetime.strptime(str(stamp)[:19], "%Y:%m:%d %H:%M:%S").isoformat()
            except ValueError:
                pass
    except Exception:  # noqa: BLE001 - unreadable metadata just means "no GPS"
        pass
    return out


# ----------------------------------------------------------------------------- geometry
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(a))


def to_local_xy(points: list[tuple[float, float]], origin: tuple[float, float]) -> np.ndarray:
    """(lat, lon) -> metres east / north of ``origin`` (accurate for field-sized areas)."""
    lat0, lon0 = origin
    k = math.cos(math.radians(lat0))
    return np.array([[math.radians(lon - lon0) * R_EARTH * k, math.radians(lat - lat0) * R_EARTH]
                     for lat, lon in points])


def polygon_area_ha(poly: list[tuple[float, float]]) -> Optional[float]:
    if len(poly) < 3:
        return None
    c = (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))
    xy = to_local_xy(poly, c)
    x, y = xy[:, 0], xy[:, 1]
    area_m2 = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area_m2 / 10_000)


def point_in_polygon(lat: float, lon: float, poly: list[tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        (y1, x1), (y2, x2) = poly[i], poly[(i + 1) % n]
        if (y1 > lat) != (y2 > lat) and lon < (x2 - x1) * (lat - y1) / (y2 - y1 + 1e-18) + x1:
            inside = not inside
    return inside


def compass(dx: float, dy: float) -> str:
    """Compass direction of a vector given in metres east (dx) and north (dy)."""
    names = ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"]
    return names[int(round(math.degrees(math.atan2(dy, dx)) / 45)) % 8]


# ----------------------------------------------------------------------------- reverse geocoding
def reverse_geocode(lat: float, lon: float) -> dict:
    """Region and country for a point (OpenStreetMap Nominatim, cached). Returns {} when offline."""
    key = hashlib.md5(f"{lat:.2f},{lon:.2f}".encode(), usedforsecurity=False).hexdigest()
    path = config.CACHE_DIR / f"geo_{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        r = requests.get("https://nominatim.openstreetmap.org/reverse",
                         params={"lat": round(lat, 2), "lon": round(lon, 2), "format": "jsonv2", "zoom": 10, "accept-language": "en"},   # ~1 km is plenty for a village name
                         headers={"User-Agent": config.USER_AGENT}, timeout=config.HTTP_TIMEOUT)
        r.raise_for_status()
        a = r.json().get("address", {})
        out = {"country": a.get("country", ""), "state": a.get("state") or a.get("region", ""),
               "county": a.get("county") or a.get("state_district", ""),
               "place": a.get("city") or a.get("town") or a.get("village") or a.get("hamlet", ""),
               "label": ", ".join(x for x in [a.get("village") or a.get("town") or a.get("city"),
                                              a.get("county") or a.get("state_district"), a.get("state"),
                                              a.get("country")] if x)}
        path.write_text(json.dumps(out), encoding="utf-8")
        return out
    except Exception:  # noqa: BLE001
        return {}
