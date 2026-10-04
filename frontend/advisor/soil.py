"""Soil facts for a field: the user's soil test when there is one, a SoilGrids estimate when there is not."""
from __future__ import annotations

import hashlib
import json
from typing import Optional

import requests

from . import config
from .schemas import SoilInput

SOILGRIDS_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
DEPTH_WEIGHTS = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15}      # topsoil to plough depth, weighted by thickness


def usda_texture(sand: float, silt: float, clay: float) -> str:
    """USDA soil texture class from percentages that sum to about 100."""
    if clay >= 40 and silt >= 40:
        return "Silty clay"
    if clay >= 40 and sand >= 45:
        return "Sandy clay"
    if clay >= 40:
        return "Clay"
    if 27 <= clay < 40 and sand <= 20:
        return "Silty clay loam"
    if 27 <= clay < 40 and sand <= 45:
        return "Clay loam"
    if 27 <= clay < 40:
        return "Sandy clay loam" if sand > 45 else "Clay loam"
    if 20 <= clay < 27 and silt < 28 and sand > 45:
        return "Sandy clay loam"
    if silt >= 80 and clay < 12:
        return "Silt"
    if silt >= 50 and (12 <= clay < 27 or (silt < 80 and clay < 12)):
        return "Silt loam"
    if 7 <= clay < 27 and 28 <= silt < 50 and sand <= 52:
        return "Loam"
    if sand >= 85 and (silt + 1.5 * clay) < 15:
        return "Sand"
    if sand >= 70 and (silt + 1.5 * clay) < 30 and sand < 90:
        return "Loamy sand"
    return "Sandy loam"


def estimate_soilgrids(lat: float, lon: float) -> Optional[dict]:
    """Rough topsoil values (0-30 cm) from the global 250 m SoilGrids map. None when the service is down."""
    key = hashlib.md5(f"{lat:.3f},{lon:.3f}".encode()).hexdigest()
    path = config.CACHE_DIR / f"soil_{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")) or None
    params = [("lon", f"{lon:.5f}"), ("lat", f"{lat:.5f}"), ("value", "mean")]
    params += [("property", p) for p in ("phh2o", "soc", "clay", "sand", "silt")]
    params += [("depth", d) for d in DEPTH_WEIGHTS]
    try:
        r = requests.get(SOILGRIDS_URL, params=params, headers={"User-Agent": config.USER_AGENT}, timeout=40)
        r.raise_for_status()
        layers = r.json()["properties"]["layers"]
        vals = {}
        for layer in layers:
            factor = layer["unit_measure"].get("d_factor", 1) or 1
            num = den = 0.0
            for dep in layer["depths"]:
                m = dep["values"].get("mean")
                w = DEPTH_WEIGHTS.get(dep["label"])
                if m is not None and w:
                    num += (m / factor) * w
                    den += w
            if den:
                vals[layer["name"]] = num / den
        if "phh2o" not in vals:
            raise ValueError("the SoilGrids map returned no data for this point (the public service is often empty)")
        out = {"ph": round(vals["phh2o"], 1)}
        if "soc" in vals:
            out["organic_matter_pct"] = round(vals["soc"] * 0.1724, 1)   # soil organic carbon g/kg -> organic matter %
        if all(k in vals for k in ("sand", "silt", "clay")):
            out["texture"] = usda_texture(vals["sand"], vals["silt"], vals["clay"])
        path.write_text(json.dumps(out), encoding="utf-8")
        return out
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200]}


def build_soil(user: SoilInput, lat: Optional[float], lon: Optional[float]) -> dict:
    """Soil block for the evidence packet. A user soil test always wins over the map estimate."""
    soil = {"source": "none", "ph": None, "organic_matter_pct": None, "texture": "Unknown",
            "p_level": user.p_level, "k_level": user.k_level, "n_level": user.n_level,
            "test_date": user.test_date.isoformat() if user.test_date else None}
    soil["field_source"] = {}
    if user.provided:
        soil.update(source="soil test", ph=user.ph, organic_matter_pct=user.organic_matter_pct, texture=user.texture)
        soil["field_source"] = {k: "soil test" for k, v in (("ph", user.ph), ("organic_matter_pct", user.organic_matter_pct),
                                                            ("texture", user.texture if user.texture != "Unknown" else None)) if v is not None}
    need_estimate = soil["ph"] is None or soil["organic_matter_pct"] is None or soil["texture"] == "Unknown"
    if need_estimate and lat is not None and lon is not None:
        est = estimate_soilgrids(lat, lon)
        if est and "error" not in est:
            filled = []
            for k in ("ph", "organic_matter_pct", "texture"):
                empty = soil[k] is None or soil[k] == "Unknown"
                if empty and k in est:
                    soil[k] = est[k]
                    soil["field_source"][k] = "SoilGrids estimate (250 m map)"
                    filled.append(k)
            if filled:
                soil["estimated_fields"] = filled
                soil["source"] = "soil test + SoilGrids estimate" if user.provided else "SoilGrids estimate (250 m map)"
        elif est and "error" in est:
            soil["estimate_error"] = est["error"]
    return soil
