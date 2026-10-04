"""The agronomy engine: plain code (no AI) that turns measurements into an evidence packet.

Every number the report may mention is created here and given an id. The language model only explains these
facts; a validator later rejects any figure that is not in the packet.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Optional

import numpy as np
from scipy import stats

from . import config, geo, knowledge
from .schemas import FieldInput


# ----------------------------------------------------------------------------- evidence ledger
class Ledger:
    """Collects evidence items (E1, E2, ...), drivers (D1, ...) and candidate actions (A1, ...)."""

    def __init__(self) -> None:
        self.evidence: list[dict] = []
        self.drivers: list[dict] = []
        self.actions: list[dict] = []

    def fact(self, label: str, value, unit: str = "", note: str = "") -> str:
        eid = f"E{len(self.evidence) + 1}"
        self.evidence.append({"id": eid, "label": label, "value": value, "unit": unit, "note": note})
        return eid

    def driver(self, label: str, severity: int, text: str, evidence: list[str]) -> str:
        did = f"D{len(self.drivers) + 1}"
        self.drivers.append({"id": did, "label": label, "severity": severity, "text": text, "evidence": evidence})
        return did

    def action(self, window: str, text: str, evidence: list[str], confirm: bool, must: bool = False) -> str:
        aid = f"A{len(self.actions) + 1}"
        self.actions.append({"id": aid, "window": window, "text": text, "evidence": evidence,
                             "confirm": confirm, "must_include": must})
        return aid


def _r(x: float, nd: int = 0):
    return int(round(x)) if nd == 0 else round(float(x), nd)


# ----------------------------------------------------------------------------- photo statistics
def aggregate(photos: list[dict], field: FieldInput) -> dict:
    """Per-photo densities -> field mean, spread and a 95% interval (sampling + counter uncertainty)."""
    used = [p for p in photos if p.get("usable", True)]
    n = len(used)
    out = {"n_photos": len(photos), "n_used": n, "calibration_factor": 1.0, "calibrated": False}

    # Calibration from hand counts the user typed in (counter tends to undercount on unfamiliar farms).
    pairs = [(p["count_raw"], p["hand_count"]) for p in used if p.get("hand_count") is not None]
    if len(pairs) >= 2 and sum(a for a, _ in pairs) > 0:
        factor = sum(b for _, b in pairs) / sum(a for a, _ in pairs)
        factor = float(min(1.5, max(0.7, factor)))
        out.update(calibration_factor=round(factor, 3), calibrated=True, calibration_photos=len(pairs))
    f = out["calibration_factor"]
    for p in used:
        p["count"] = p["count_raw"] * f
        p["density_m2"] = (p["count"] / field.photo_area_m2) if field.photo_area_m2 else None

    counts = np.array([p["count"] for p in used], dtype=float)
    if n == 0:
        return out
    out.update(heads_per_photo_mean=float(counts.mean()))
    if not field.photo_area_m2:
        return out

    dens = np.array([p["density_m2"] for p in used], dtype=float)
    mean = float(dens.mean())
    sd = float(dens.std(ddof=1)) if n > 1 else 0.0
    cv = (sd / mean * 100) if mean > 0 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    tcrit = float(stats.t.ppf(0.975, n - 1)) if n > 1 else 0.0
    counter_err = config.COUNTER_ERROR_CALIBRATED if out["calibrated"] else config.COUNTER_ERROR_UNCALIBRATED
    half = math.sqrt((tcrit * se) ** 2 + (counter_err * mean) ** 2)
    out.update(density_mean=mean, density_sd=sd, cv_pct=cv, ci_low=max(0.0, mean - half), ci_high=mean + half,
               counter_error=counter_err)
    return out


def spatial_pattern(photos: list[dict]) -> Optional[dict]:
    """Where the stand is thin or thick, from photo positions. Needs at least 6 photos with GPS."""
    pts = [p for p in photos if p.get("usable", True) and p.get("lat") is not None and p.get("density_m2") is not None]
    if len(pts) < 4:
        return None
    origin = (float(np.mean([p["lat"] for p in pts])), float(np.mean([p["lon"] for p in pts])))
    xy = geo.to_local_xy([(p["lat"], p["lon"]) for p in pts], origin)
    d = np.array([p["density_m2"] for p in pts])
    order = np.argsort(d)
    low = [{"name": pts[i]["name"], "density": _r(d[i])} for i in order[:3]]
    out = {"lowest": low, "gradient": None}
    if len(pts) >= 6 and d.mean() > 0:
        A = np.column_stack([np.ones(len(pts)), xy[:, 0], xy[:, 1]])
        coef, *_ = np.linalg.lstsq(A, d, rcond=None)
        pred = A @ coef
        ss_res, ss_tot = float(((d - pred) ** 2).sum()), float(((d - d.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        gx, gy = float(coef[1]), float(coef[2])
        g = math.hypot(gx, gy)
        if g > 0:
            ux, uy = gx / g, gy / g
            proj = xy @ np.array([ux, uy])
            extent = float(proj.max() - proj.min())
            change_pct = g * extent / float(d.mean()) * 100
            if r2 >= 0.35 and change_pct >= 15:
                out["gradient"] = {"toward": geo.compass(gx, gy), "away_from": geo.compass(-gx, -gy),
                                   "change_pct": _r(change_pct), "r2": round(r2, 2)}
    return out


def yield_range(agg: dict, field: FieldInput) -> Optional[dict]:
    """t/ha from heads per m2 x kernels per head x thousand-kernel weight, with low / central / high."""
    if "density_mean" not in agg:
        return None
    k_lo, k_c, k_hi = config.KERNELS_PER_HEAD
    t_lo, t_c, t_hi = config.TKW_G
    user_k, user_t = field.kernels_per_head, field.tkw_g
    if user_k:
        k_lo, k_c, k_hi = user_k * 0.9, user_k, user_k * 1.1
    if user_t:
        t_lo, t_c, t_hi = user_t * 0.95, user_t, user_t * 1.05

    def y(d, k, t):
        return d * k * t / 100_000

    res = {"low": y(agg["ci_low"], k_lo, t_lo), "central": y(agg["density_mean"], k_c, t_c),
           "high": y(agg["ci_high"], k_hi, t_hi), "kernels": (k_lo, k_c, k_hi), "tkw": (t_lo, t_c, t_hi),
           "kernels_from_user": bool(user_k), "tkw_from_user": bool(user_t)}
    return res


# ----------------------------------------------------------------------------- the analysis
def analyze(field: FieldInput, photos: list[dict], location: dict, weather: Optional[dict], soil: dict,
            gaps: list[dict]) -> dict:
    """Build the full evidence packet: facts, drivers, candidate actions, verdict and confidence."""
    L = Ledger()
    stage = field.growth_stage
    days = (date.today() - field.sowing_date).days
    agg = aggregate(photos, field)
    yld = yield_range(agg, field)
    spat = spatial_pattern(photos)
    assumptions: list[str] = []
    ev: dict[str, str] = {}

    # --- context facts
    ev["stage"] = L.fact("Growth stage reported", stage, "", f"{days} days after sowing on {field.sowing_date.isoformat()}")
    if location.get("label"):
        ev["region"] = L.fact("Region", location["label"], "", location.get("how", ""))
    area = field.area_ha or location.get("area_ha")
    if area:
        ev["area"] = L.fact("Field area", _r(area, 1), "ha", "from the drawn boundary" if location.get("area_ha") else "entered")

    # --- stand density
    heads_known = stage in config.HEADS_VISIBLE_STAGES
    ref_low = field.reference_heads_low if field.reference_heads_low else config.HEADS_REFERENCE[0]
    ref_high = field.reference_heads_high if field.reference_heads_high else config.HEADS_REFERENCE[1]
    ref_is_placeholder = not (field.reference_heads_low and field.reference_heads_high)
    if "density_mean" in agg:
        ci = f"95% interval {_r(agg['ci_low'])} to {_r(agg['ci_high'])}"
        if agg["calibrated"]:
            ci += f"; counts scaled by {agg['calibration_factor']} from {agg['calibration_photos']} hand-counted photos"
        ev["density"] = L.fact("Head density", _r(agg["density_mean"]), "heads/m²", ci)
        ev["ref"] = L.fact("Reference density band", f"{_r(ref_low)} to {_r(ref_high)}", "heads/m²",
                           "generic placeholder, not a local value" if ref_is_placeholder else "entered by the user")
        if ref_is_placeholder:
            assumptions.append("The reference head-density band is a generic placeholder. Replace it with a local value.")
        if agg["n_used"] >= 2:
            ev["cv"] = L.fact("Variation between photos", _r(agg["cv_pct"]), "%",
                              f"{agg['n_used']} photos; lower means a more even stand")
    elif agg.get("n_used"):
        ev["heads"] = L.fact("Heads per photo (average)", _r(agg["heads_per_photo_mean"], 1), "heads",
                             "no photo area was given, so no heads per m² or yield")
        assumptions.append("Without the ground area of each photo the counts cannot become heads per m² or a yield.")
    if agg.get("n_used"):
        ev["photos"] = L.fact("Photos analysed", agg["n_used"], "photos",
                              f"{agg['n_photos'] - agg['n_used']} skipped for quality" if agg["n_photos"] != agg["n_used"] else "")

    # --- yield
    if yld:
        ev["yield"] = L.fact("Yield forecast (central)", _r(yld["central"], 1), "t/ha",
                             f"range {_r(yld['low'], 1)} to {_r(yld['high'], 1)} t/ha")
        k = yld["kernels"]; t = yld["tkw"]
        assumptions.append(
            f"Yield assumes {_r(k[0])} to {_r(k[2])} kernels per head" + (" (entered)" if yld["kernels_from_user"] else " (generic default)")
            + f" and {_r(t[0])} to {_r(t[2])} g per 1000 kernels" + (" (entered)." if yld["tkw_from_user"] else " (generic default)."))
        if area:
            tot = yld["central"] * area
            ev["total"] = L.fact("Whole-field production (central)", _r(tot, 1), "t",
                                 f"range {_r(yld['low'] * area, 1)} to {_r(yld['high'] * area, 1)} t over {_r(area, 1)} ha")
        if field.target_yield_t_ha:
            ev["target"] = L.fact("Target yield entered", _r(field.target_yield_t_ha, 1), "t/ha", "")

    # --- spatial pattern
    if spat:
        low = ", ".join(f"{x['name']} ({x['density']} heads/m²)" for x in spat["lowest"])
        ev["lowest"] = L.fact("Thinnest photo points", low, "", "")
        if spat["gradient"]:
            g = spat["gradient"]
            ev["gradient"] = L.fact("Spatial trend in head density", f"thinner toward the {g['away_from']}", "",
                                    f"about {g['change_pct']}% change across the sampled area (fit R² {g['r2']})")

    # --- weather
    wx = (weather or {}).get("summary") if weather and "summary" in weather else None
    if wx:
        ev["gdd"] = L.fact("Growing degree days since sowing", wx["gdd_since_sowing"], "°C·days", "base 0 °C")
        ev["rain"] = L.fact("Rain since sowing", wx["rain_since_sowing_mm"], "mm", f"data to {wx['last_date']}")
        ev["balance"] = L.fact("Rain minus crop water use, last 4 weeks", wx["water_balance_28d_mm"], "mm",
                               f"rain {wx['rain_28d_mm']} mm, reference evapotranspiration {wx['et0_28d_mm']} mm")
        ev["heat"] = L.fact("Hot days (30 °C or more), last 3 weeks", wx["heat_days_21d"], "days",
                            f"peak {wx['tmax_peak_21d']} °C" if wx.get("tmax_peak_21d") is not None else "")
        ev["wet"] = L.fact("Rainy days (1 mm or more), last 2 weeks", wx["rainy_days_14d"], "days", "")

    # --- soil
    fsrc = soil.get("field_source", {})

    def src(key):
        s = fsrc.get(key, soil["source"])
        return s + (f", tested {soil['test_date']}" if s == "soil test" and soil.get("test_date") else "")

    if soil.get("ph") is not None:
        ev["ph"] = L.fact("Soil pH", soil["ph"], "", src("ph"))
    if soil.get("organic_matter_pct") is not None:
        ev["om"] = L.fact("Soil organic matter", soil["organic_matter_pct"], "%", src("organic_matter_pct"))
    if soil.get("texture") not in (None, "Unknown"):
        ev["texture"] = L.fact("Soil texture", soil["texture"], "", src("texture"))
    for key, lab in (("p_level", "Phosphorus"), ("k_level", "Potassium"), ("n_level", "Nitrogen")):
        if soil.get(key) not in (None, "Unknown"):
            ev[key] = L.fact(f"{lab} level (lab rating)", soil[key], "", "as rated on the soil test")
    ph_est = fsrc.get("ph", "").startswith("SoilGrids")
    om_est = fsrc.get("organic_matter_pct", "").startswith("SoilGrids")
    est = ph_est

    # =========================================================== drivers (what makes the field good or poor)
    def drv(label, sev, text, keys):
        L.driver(label, sev, text, [ev[k] for k in keys if k in ev])

    if "density_mean" in agg and heads_known:
        m = agg["density_mean"]
        if m < ref_low * 0.85:
            drv("Thin stand", 2, f"Head density is well below the reference band of {_r(ref_low)} to {_r(ref_high)} heads/m².", ["density", "ref"])
        elif m < ref_low:
            drv("Slightly thin stand", 1, f"Head density is a little below the reference band of {_r(ref_low)} to {_r(ref_high)} heads/m².", ["density", "ref"])
        elif m > ref_high * 1.15:
            drv("Very dense stand", 0, "Head density is above the reference band, which can raise lodging risk.", ["density", "ref"])
        else:
            drv("Stand density in range", 0, "Head density is within the reference band.", ["density", "ref"])
    if "cv" in ev:
        cv = agg["cv_pct"]
        if cv >= config.CV_HIGH:
            drv("Uneven stand", 2, f"Head density varies a lot between photo points ({_r(cv)}%).", ["cv", "gradient", "lowest"])
        elif cv >= config.CV_MODERATE:
            drv("Somewhat uneven stand", 1, f"Head density varies moderately between photo points ({_r(cv)}%).", ["cv", "gradient", "lowest"])
    if wx and stage in ("Flowering", "Grain fill", "Heading"):
        bal = wx["water_balance_28d_mm"]
        if bal <= -60:
            sev = 1 if field.irrigated else 2
            drv("Water shortage", sev, f"Rain is {_r(-bal)} mm short of crop water use over the last 4 weeks.", ["balance"])
        elif bal <= -30:
            drv("Some water shortage", 1, f"Rain is {_r(-bal)} mm short of crop water use over the last 4 weeks.", ["balance"])
    if wx and stage in ("Flowering", "Grain fill"):
        hd = wx["heat_days_21d"]
        if hd >= 8:
            drv("Heat stress", 2, f"{hd} hot days (30 °C or more) in the last 3 weeks during a heat-sensitive stage.", ["heat"])
        elif hd >= 4:
            drv("Some heat stress", 1, f"{hd} hot days (30 °C or more) in the last 3 weeks.", ["heat"])
    if wx and stage in ("Heading", "Flowering", "Grain fill") and wx["rainy_days_14d"] >= 7:
        drv("Wet spell", 1, f"{wx['rainy_days_14d']} rainy days in the last 2 weeks raise the risk of head diseases.", ["wet", "stage"])
    if soil.get("ph") is not None:
        ph = soil["ph"]
        cap = 1 if est else 2
        if ph < 5.5:
            drv("Acid soil", cap, f"Soil pH {ph} is acid" + (" (map estimate)." if est else "."), ["ph"])
        elif ph < 6.0:
            drv("Slightly acid soil", 1, f"Soil pH {ph} is slightly acid" + (" (map estimate)." if est else "."), ["ph"])
        elif ph > 7.8:
            drv("Alkaline soil", 1, f"Soil pH {ph} is alkaline, which can limit some micronutrients.", ["ph"])
    if soil.get("organic_matter_pct") is not None and soil["organic_matter_pct"] < 2.0:
        drv("Low organic matter", 1, f"Organic matter is low at {soil['organic_matter_pct']}%" + (" (map estimate)." if om_est else "."), ["om"])
    low_nutrients = [lab for key, lab in (("p_level", "phosphorus"), ("k_level", "potassium"), ("n_level", "nitrogen"))
                     if soil.get(key) == "Low"]
    if low_nutrients:
        drv("Low nutrient rating", 1, f"The soil test rates {', '.join(low_nutrients)} as low.", [k for k in ("p_level", "k_level", "n_level") if soil.get(k) == "Low"])
    if yld and field.target_yield_t_ha and yld["central"] < 0.85 * field.target_yield_t_ha:
        drv("Forecast below target", 1, "The central yield forecast is more than 15% below the target entered.", ["yield", "target"])
    prev = field.previous_crop.lower()
    if any(w in prev for w in ("wheat", "barley", "triticale", "rye", "cereal")):
        drv("Cereal after cereal", 0, "The previous crop was a cereal, which carries disease over to this crop.", ["stage"])

    # =========================================================== candidate actions
    D = {d["label"]: d for d in L.drivers}

    def ids(*labels):
        out = []
        for lab in labels:
            out += D[lab]["evidence"] if lab in D else []
        return sorted(set(out), key=lambda s: int(s[1:]))

    if yld:
        L.action("this_season", "Use the yield range for harvest planning, storage and marketing, and count again nearer maturity.",
                 [ev[k] for k in ("yield", "total") if k in ev], confirm=False)
    if "Water shortage" in D or "Some water shortage" in D:
        lab = "Water shortage" if "Water shortage" in D else "Some water shortage"
        if field.irrigated:
            L.action("this_season", "Irrigate through grain fill while the water shortage lasts.", ids(lab), confirm=False, must=True)
        else:
            L.action("this_season", "Water cannot be added to a rain-fed crop. Lower the yield expectation toward the low end of the range.", ids(lab), confirm=False)
    if "Wet spell" in D:
        L.action("this_season", "Walk the field to look for head diseases. A licensed local advisor decides whether any treatment is worth it, and which product and timing the label allows.", ids("Wet spell"), confirm=True, must=True)
    if "Heat stress" in D or "Some heat stress" in D:
        L.action("this_season", "Expect fewer or lighter grains after the heat. Count again later and treat the low end of the range as more likely.", ids("Heat stress" if "Heat stress" in D else "Some heat stress"), confirm=False)
    if "Uneven stand" in D or "Somewhat uneven stand" in D:
        lab = "Uneven stand" if "Uneven stand" in D else "Somewhat uneven stand"
        L.action("this_season", "Walk the thinnest photo points and look for the cause: waterlogging, compaction, weeds, pests or disease. Note what you find with a photo.",
                 ids(lab), confirm=False, must=(lab == "Uneven stand"))
        L.action("next_season", "Check drainage, compaction and sowing quality in the weak zones, and consider managing them as a separate zone.", ids(lab), confirm=True)
    if "Thin stand" in D or "Slightly thin stand" in D:
        lab = "Thin stand" if "Thin stand" in D else "Slightly thin stand"
        L.action("next_season", "Review seed rate, sowing date, seedbed and early nitrogen timing against what an agronomist expects for your area.", ids(lab), confirm=True)
    if "Acid soil" in D or "Slightly acid soil" in D:
        lab = "Acid soil" if "Acid soil" in D else "Slightly acid soil"
        L.action("next_season", "Test the soil pH in each zone and correct acidity with lime where local guidance says it pays. Retest afterwards.", ids(lab), confirm=True, must=("Acid soil" in D))
    if "Alkaline soil" in D:
        L.action("next_season", "Ask an agronomist about micronutrient availability (for example manganese and zinc) on this alkaline soil.", ids("Alkaline soil"), confirm=True)
    if "Low organic matter" in D:
        L.action("next_season", "Build organic matter over time with crop residues, manure or cover crops where they fit the rotation.", ids("Low organic matter"), confirm=False)
    if "Low nutrient rating" in D:
        L.action("next_season", "Plan phosphorus, potassium or nitrogen for the next crop from the soil test and a yield goal, using your local fertilizer recommendation.", ids("Low nutrient rating"), confirm=True, must=True)
    if "Cereal after cereal" in D:
        L.action("next_season", "Break the run of cereals with a different crop where the rotation allows.", ids("Cereal after cereal"), confirm=False)
    L.action("next_season", "Set the nitrogen rate and its timing from a spring soil test, the crop that came before and a realistic yield goal, with your local guideline.",
             [], confirm=True)

    # =========================================================== verdict + confidence
    majors = sum(1 for d in L.drivers if d["severity"] >= 2)
    total = sum(d["severity"] for d in L.drivers)
    if not heads_known:
        label = "Not enough information"
    elif majors >= 2 or total >= 5:
        label = "Poor"
    elif majors == 1 or total >= 2:
        label = "Watch"
    else:
        label = "Good"

    conf_pts, reasons = 100, []
    if not heads_known:
        conf_pts -= 50; reasons.append("heads are not visible at this stage")
    if agg.get("n_used", 0) < 5:
        conf_pts -= 25; reasons.append(f"only {agg.get('n_used', 0)} usable photos")
    elif agg["n_used"] < 10:
        conf_pts -= 10; reasons.append(f"{agg['n_used']} photos (10 or more is better)")
    if not field.photo_area_m2:
        conf_pts -= 25; reasons.append("photo ground area unknown")
    if not wx:
        conf_pts -= 15; reasons.append("no weather data")
    if not field.soil.provided:
        conf_pts -= 10; reasons.append("no soil test" + (" (map estimate used)" if (ph_est or om_est) else ""))
    if location.get("confidence") == "low":
        conf_pts -= 10; reasons.append("location not confirmed")
    if agg.get("calibrated"):
        conf_pts += 5
    conf_label = "High" if conf_pts >= 85 else "Medium" if conf_pts >= 55 else "Low"

    assumptions.append("Reference bands, kernel counts and soil thresholds are generic and should be replaced with local values.")
    assumptions.append("Advice is decision support. Confirm fertilizer and spray decisions with a local agronomist.")

    open_q = []
    if not field.inputs_applied.strip():
        open_q.append("What fertilizer and sprays have been applied so far this season?")
    if not field.soil.provided:
        open_q.append("Is a recent soil test available for this field?")
    if "Uneven stand" in D or "Somewhat uneven stand" in D:
        open_q.append("Was there waterlogging, compaction or a sowing problem in the thinnest parts of the field?")
    if not field.kernels_per_head or not field.tkw_g:
        open_q.append("Do you know the usual kernels per head and kernel weight for this variety?")

    packet = {
        "field": {"name": field.name, "crop": field.crop, "variety": field.variety or "not given",
                  "stage": stage, "days_since_sowing": days, "irrigated": field.irrigated,
                  "region": location.get("label") or "unknown", "area_ha": _r(area, 1) if area else None,
                  "previous_crop": field.previous_crop or "not given", "inputs_applied": field.inputs_applied or "not given",
                  "problems_noticed": field.problems_noticed or "none given"},
        "verdict": {"label": label, "confidence": conf_label, "confidence_reasons": reasons},
        "stage_note": knowledge.stage_note(stage),
        "evidence": L.evidence, "drivers": L.drivers, "candidate_actions": L.actions,
        "assumptions": assumptions, "open_questions": open_q, "gaps": [g["message"] for g in gaps],
    }
    return {"packet": packet, "aggregate": agg, "yield": yld, "spatial": spat, "verdict": label,
            "confidence": conf_label, "confidence_points": conf_pts,
            "needs_review": conf_label == "Low" or label == "Poor"}
