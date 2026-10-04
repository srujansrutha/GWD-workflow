"""The field-report workflow as a LangGraph graph.

  START -> intake -> detect_photos -> locate -> (weather | soil in parallel) -> find_gaps -> analyze
        -> write_report -> validate_report -> finalize
                                   |-> write_report (retry once with the error list)
                                   |-> fallback_report -> finalize

Any step that cannot continue sets status "needs_input" and the run ends early with questions for the user.
"""
from __future__ import annotations

import io
import statistics
import time
from datetime import date
from typing import Annotated, Optional, TypedDict

import cv2
import numpy as np
from langgraph.graph import END, START, StateGraph
from langchain_core.runnables import RunnableConfig
from PIL import Image

import wheat_detector as wd

from . import config as cfg, engine, geo, llm, soil as soilmod, weather as wxmod
from .schemas import FieldInput, PhotoIn, ReportDraft, RunStats

MAX_LLM_ATTEMPTS = 2
BLUR_WARN, BLUR_BAD = 30.0, 10.0       # Laplacian variance at 1024 px; ordinary photos score 200 to 500
DARK, BRIGHT = 25.0, 200.0


def _merge(a: Optional[dict], b: Optional[dict]) -> dict:
    return {**(a or {}), **(b or {})}


def _append(a: Optional[list], b: Optional[list]) -> list:
    return (a or []) + (b or [])


class AdvisorState(TypedDict, total=False):
    field: FieldInput
    photos: list[PhotoIn]
    status: str                                   # running | needs_input | done
    issues: Annotated[list, _append]              # {level: error|warning, message}
    photo_results: list[dict]
    location: dict
    weather: Optional[dict]
    soil: dict
    gaps: list[dict]
    analysis: dict
    draft: Optional[ReportDraft]
    validation_errors: list[str]
    attempts: int
    stats: RunStats
    llm_info: Annotated[list, _append]
    timings: Annotated[dict, _merge]
    report: dict


def timed(name):
    """Record how long a node took, even when nodes run in parallel."""
    def deco(fn):
        def wrapper(state, config: RunnableConfig = None):
            t = time.time()
            out = fn(state, config) if fn.__code__.co_argcount > 1 else fn(state)
            out = dict(out or {})
            out["timings"] = {name: round(time.time() - t, 2)}
            return out
        wrapper.__name__ = fn.__name__
        return wrapper
    return deco


# ----------------------------------------------------------------------------- 1. intake
@timed("intake")
def intake(state: AdvisorState) -> dict:
    f: FieldInput = state["field"]
    issues = []
    if not state.get("photos"):
        issues.append({"level": "error", "message": "Upload at least one photo of the crop."})
    if f.sowing_date > date.today():
        issues.append({"level": "error", "message": "The sowing date is in the future."})
    elif (date.today() - f.sowing_date).days > 450:
        issues.append({"level": "error", "message": "The sowing date is more than 450 days ago. Check the year."})
    if f.growth_stage not in cfg.HEADS_VISIBLE_STAGES:
        issues.append({"level": "error", "message": (
            f"At the {f.growth_stage.lower()} stage wheat heads are not visible yet, so a head count would mean little. "
            "Come back from heading, or correct the growth stage if it is wrong.")})
    if f.reference_heads_low and f.reference_heads_high and f.reference_heads_low > f.reference_heads_high:
        issues.append({"level": "error", "message": "The reference head density low value is above the high value."})
    blocked = any(i["level"] == "error" for i in issues)
    return {"issues": issues, "status": "needs_input" if blocked else "running"}


# ----------------------------------------------------------------------------- 2. detect
def _quality(img: Image.Image) -> dict:
    g = cv2.cvtColor(np.asarray(img.resize((1024, 1024))), cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
    bright = float(g.mean())
    flags = []
    if blur < BLUR_BAD:
        flags.append("very blurry")
    elif blur < BLUR_WARN:
        flags.append("blurry")
    if bright < DARK:
        flags.append("too dark")
    elif bright > BRIGHT:
        flags.append("overexposed")
    return {"blur": round(blur, 1), "brightness": round(bright, 1), "flags": flags}


@timed("detect_photos")
def detect_photos(state: AdvisorState, config: RunnableConfig) -> dict:
    detector: wd.Detector = config["configurable"]["detector"]
    f: FieldInput = state["field"]
    results, issues = [], []
    for p in state["photos"]:
        exif = geo.read_exif(p.raw)
        r = {"name": p.name, "usable": False, "hand_count": p.hand_count, **exif}
        try:
            dec = wd.decode_image(p.raw)
        except wd.ImageReadError as e:
            r["error"] = str(e)
            issues.append({"level": "warning", "message": f"{p.name}: {e} It was skipped."})
            results.append(r)
            continue
        det = detector.detect(dec.image)
        boxes, scores = det.at(f.conf)
        q = _quality(dec.image)
        small = wd.annotate(dec.image, boxes, scores, color="#22c55e", thickness=1.5)
        scale = min(1.0, 1500 / max(small.size))
        if scale < 1:
            small = small.resize((round(small.width * scale), round(small.height * scale)), Image.LANCZOS)
        r.update(usable=True, count_raw=int(len(boxes)), mean_conf=float(scores.mean()) if len(scores) else 0.0,
                 width=dec.width, height=dec.height, fmt=dec.fmt, ms=round(det.ms), quality=q,
                 annotated=wd.to_jpeg(small, 88))
        if q["flags"]:
            issues.append({"level": "warning", "message": f"{p.name} looks {' and '.join(q['flags'])}. Its count may be off."})
        results.append(r)
    if not any(r["usable"] for r in results):
        issues.append({"level": "error", "message": "None of the uploaded files could be read as an image."})
        return {"photo_results": results, "issues": issues, "status": "needs_input"}
    return {"photo_results": results, "issues": issues}


# ----------------------------------------------------------------------------- 3. locate
@timed("locate")
def locate(state: AdvisorState) -> dict:
    f: FieldInput = state["field"]
    photos = state["photo_results"]
    gps = [p for p in photos if p.get("has_gps")]
    loc: dict = {"lat": f.lat, "lon": f.lon, "n_gps": len(gps), "warnings": []}
    issues = []

    if f.lat is not None and f.lon is not None:
        loc["how"] = "pin entered by the user"
    elif gps:
        loc["lat"] = float(statistics.median(p["lat"] for p in gps))
        loc["lon"] = float(statistics.median(p["lon"] for p in gps))
        loc["how"] = f"median of GPS in {len(gps)} photo{'s' if len(gps) != 1 else ''}"
    else:
        issues.append({"level": "error", "message": (
            "I could not find the field. Drop a pin on the map, or upload original photos that still carry GPS "
            "(messaging apps remove it).")})
        return {"location": loc, "issues": issues, "status": "needs_input"}

    lat, lon = loc["lat"], loc["lon"]
    if gps:
        d = [geo.haversine_m(lat, lon, p["lat"], p["lon"]) for p in gps]
        loc["spread_m"] = round(float(max(d)))
        loc["median_dist_m"] = round(float(statistics.median(d)))
        for p, dist in zip(gps, d):
            p["dist_from_pin_m"] = round(dist)
        if f.lat is not None and statistics.median(d) > 2000:
            loc["warnings"].append("Photo GPS is more than 2 km from the pin. Check the pin or the photos.")
    poly = f.polygon
    if len(poly) >= 3:
        loc["area_ha"] = geo.polygon_area_ha(poly)
        for p in gps:
            p["inside_field"] = geo.point_in_polygon(p["lat"], p["lon"], poly)
        loc["n_inside"] = sum(1 for p in gps if p.get("inside_field"))
        outside = [p["name"] for p in gps if not p.get("inside_field")]
        if outside:
            loc["warnings"].append(f"{len(outside)} photo{'s were' if len(outside) != 1 else ' was'} taken outside the drawn boundary: {', '.join(outside[:3])}.")
    # confidence in the location
    inside_ok = len(poly) >= 3 and gps and loc["n_inside"] / len(gps) >= 0.8
    agree = len(gps) >= 3 and loc.get("spread_m", 1e9) <= 1500
    loc["confidence"] = "high" if (inside_ok or (f.lat is not None and agree)) else ("medium" if (f.lat is not None or gps) else "low")
    loc["accuracy_note"] = ("Phone GPS is usually within about 5 m in the open and drifts under canopy."
                            if gps else "No GPS in the photos; the position is the pin only.")
    rg = geo.reverse_geocode(lat, lon)
    loc.update({k: v for k, v in rg.items()})
    for w in loc["warnings"]:
        issues.append({"level": "warning", "message": w})
    return {"location": loc, "issues": issues}


def after_locate(state: AdvisorState):
    return END if state.get("status") == "needs_input" else ["weather", "soil"]


# ----------------------------------------------------------------------------- 4. enrich (parallel)
@timed("weather")
def weather_node(state: AdvisorState) -> dict:
    f, loc = state["field"], state["location"]
    return {"weather": wxmod.get_weather(loc["lat"], loc["lon"], f.sowing_date, f.growth_stage, f.irrigated)}


@timed("soil")
def soil_node(state: AdvisorState) -> dict:
    f, loc = state["field"], state["location"]
    return {"soil": soilmod.build_soil(f.soil, loc["lat"], loc["lon"])}


# ----------------------------------------------------------------------------- 5. gaps
@timed("find_gaps")
def find_gaps(state: AdvisorState) -> dict:
    f, loc, wx, soil = state["field"], state["location"], state.get("weather"), state["soil"]
    used = [p for p in state["photo_results"] if p["usable"]]
    gaps = []
    if not f.photo_area_m2:
        gaps.append({"level": "important", "message": "Give the ground area each photo covers (for example 0.25 m² for a 50 × 50 cm frame) to turn counts into heads per m² and a yield."})
    if len(used) < 10:
        gaps.append({"level": "important" if len(used) < 5 else "minor", "message": f"Only {len(used)} photo{'s' if len(used) != 1 else ''} analysed. Ten or more spread over the field give a much better field average."})
    if not wx or "error" in wx:
        gaps.append({"level": "important", "message": "Weather data could not be loaded, so water and heat checks were skipped." + (f" ({wx['error']})" if wx and 'error' in wx else "")})
    if not f.soil.provided:
        msg = "No soil test was entered."
        if soil.get("source", "").startswith("SoilGrids"):
            msg += " Values come from a coarse 250 m map and are only a guide."
        elif soil.get("estimate_error"):
            msg += " The soil map service had no data for this point, so soil advice was skipped."
        gaps.append({"level": "minor", "message": msg})
    if len(f.polygon) < 3 and not f.area_ha:
        gaps.append({"level": "minor", "message": "Draw the field boundary (or enter the area) to get a whole-field production estimate."})
    if loc.get("confidence") == "low":
        gaps.append({"level": "important", "message": "The location could not be confirmed."})
    return {"gaps": gaps}


# ----------------------------------------------------------------------------- 6. analyze
@timed("analyze")
def analyze(state: AdvisorState) -> dict:
    out = engine.analyze(state["field"], state["photo_results"], state["location"], state.get("weather"),
                         state["soil"], state.get("gaps", []))
    return {"analysis": out, "attempts": 0, "validation_errors": [], "stats": RunStats(model=cfg.OLLAMA_MODEL)}


# ----------------------------------------------------------------------------- 7. write + validate
@timed("write_report")
def write_report(state: AdvisorState) -> dict:
    packet = state["analysis"]["packet"]
    stats: RunStats = state["stats"]
    if cfg.FORCE_FALLBACK:
        return {"draft": None, "attempts": MAX_LLM_ATTEMPTS, "validation_errors": ["language model disabled"]}
    if state.get("attempts", 0) == 0:
        s = llm.status()
        if not (s["server"] and s["installed"]):
            why = "Ollama is not running" if not s["server"] else f"model {cfg.OLLAMA_MODEL} is not installed"
            stats.errors.append(why)
            return {"draft": None, "attempts": MAX_LLM_ATTEMPTS, "validation_errors": [why], "stats": stats}
    draft, info = llm.call_model(packet, state.get("validation_errors") or None)
    stats.attempts += 1
    stats.seconds += info["seconds"]
    stats.prompt_tokens += info["prompt_tokens"]
    stats.output_tokens += info["output_tokens"]
    if info["error"]:
        stats.errors.append(info["error"])
    return {"draft": draft, "attempts": state.get("attempts", 0) + 1, "stats": stats, "llm_info": [info]}


@timed("validate_report")
def validate_report(state: AdvisorState) -> dict:
    draft = state.get("draft")
    if draft is None:
        errs = state.get("validation_errors") or ["no usable reply from the model"]
        if state.get("llm_info") and state["llm_info"][-1].get("error"):
            errs = [state["llm_info"][-1]["error"]]
        return {"validation_errors": errs}
    errs = llm.validate(draft, state["analysis"]["packet"])
    stats: RunStats = state["stats"]
    if errs:
        stats.errors.extend(errs)
    return {"validation_errors": errs, "stats": stats}


def after_validate(state: AdvisorState) -> str:
    if not state.get("validation_errors"):
        return "finalize"
    if state.get("attempts", 0) < MAX_LLM_ATTEMPTS:
        return "write_report"
    return "fallback_report"


@timed("fallback_report")
def fallback_report(state: AdvisorState) -> dict:
    stats: RunStats = state["stats"]
    stats.used_fallback = True
    return {"draft": llm.fallback_draft(state["analysis"]["packet"]), "stats": stats}


# ----------------------------------------------------------------------------- 8. finalize
@timed("finalize")
def finalize(state: AdvisorState) -> dict:
    a, f = state["analysis"], state["field"]
    packet = a["packet"]
    agg = a["aggregate"]
    stats: RunStats = state["stats"]
    ev = {e["id"]: e for e in packet["evidence"]}
    act = {x["id"]: x for x in packet["candidate_actions"]}
    draft: ReportDraft = state["draft"]
    actions = []
    for n in draft.actions:
        base = act.get(n.action_id)
        if base:
            actions.append({"id": n.action_id, "window": base["window"], "text": base["text"], "why": n.explanation,
                            "evidence_ids": n.evidence_ids or base["evidence"], "confirm": base["confirm"]})
    photos = [{k: v for k, v in p.items() if k != "raw"} for p in state["photo_results"]]
    report = {
        "field": f.model_dump(mode="json", exclude={"soil"}), "verdict": a["verdict"], "confidence": a["confidence"],
        "needs_review": a["needs_review"], "summary": draft.summary, "reasons": [r.model_dump() for r in draft.reasons],
        "actions": actions, "open_questions": draft.open_questions, "limitations": draft.limitations,
        "assumptions": packet["assumptions"], "gaps": state.get("gaps", []), "issues": state.get("issues", []),
        "evidence": ev, "drivers": packet["drivers"], "packet": packet, "aggregate": agg, "yield": a["yield"],
        "spatial": a["spatial"], "photos": photos, "location": state["location"], "weather": state.get("weather"),
        "soil": state["soil"], "stats": stats.__dict__, "timings": state.get("timings", {}),
        "generated": date.today().isoformat(),
    }
    return {"report": report, "status": "done"}


# ----------------------------------------------------------------------------- wiring
def build_graph():
    g = StateGraph(AdvisorState)
    for name, fn in (("intake", intake), ("detect_photos", detect_photos), ("locate", locate), ("weather", weather_node),
                     ("soil", soil_node), ("find_gaps", find_gaps), ("analyze", analyze), ("write_report", write_report),
                     ("validate_report", validate_report), ("fallback_report", fallback_report), ("finalize", finalize)):
        g.add_node(name, fn)
    g.add_edge(START, "intake")
    g.add_conditional_edges("intake", lambda s: END if s.get("status") == "needs_input" else "detect_photos",
                            {END: END, "detect_photos": "detect_photos"})
    g.add_conditional_edges("detect_photos", lambda s: END if s.get("status") == "needs_input" else "locate",
                            {END: END, "locate": "locate"})
    g.add_conditional_edges("locate", after_locate, ["weather", "soil", END])
    g.add_edge(["weather", "soil"], "find_gaps")
    g.add_edge("find_gaps", "analyze")
    g.add_edge("analyze", "write_report")
    g.add_edge("write_report", "validate_report")
    g.add_conditional_edges("validate_report", after_validate, ["finalize", "write_report", "fallback_report"])
    g.add_edge("fallback_report", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


NODE_LABELS = {
    "intake": "Checking your inputs", "detect_photos": "Counting wheat heads on the GPU",
    "locate": "Locating the field", "weather": "Fetching weather", "soil": "Looking up soil",
    "find_gaps": "Looking for missing information", "analyze": "Analyzing the field",
    "write_report": "Writing the report with the local model", "validate_report": "Checking the report against the evidence",
    "fallback_report": "Using the safe rule-based report", "finalize": "Putting the report together",
}
