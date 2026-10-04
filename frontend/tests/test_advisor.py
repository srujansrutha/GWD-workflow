"""Tests for the field advisor. No GPU, network or language model needed.

Run from the frontend folder:   ..\\.venv\\Scripts\\python -m pytest tests -q
"""
from __future__ import annotations

import io
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["ADVISOR_FORCE_FALLBACK"] = "1"          # never call Ollama in tests

import wheat_detector as wd                         # noqa: E402
from advisor import config, engine, geo, graph, llm, soil, weather  # noqa: E402
from advisor.schemas import ActionNote, FieldInput, PhotoIn, Reason, ReportDraft, SoilInput  # noqa: E402

POLY = [(-35.1000, 147.3700), (-35.1000, 147.3750), (-35.1040, 147.3750), (-35.1040, 147.3700)]
WX = {"summary": {"days": 140, "gdd_since_sowing": 1600, "rain_since_sowing_mm": 280, "rain_28d_mm": 19, "et0_28d_mm": 95,
                  "water_balance_28d_mm": -76, "heat_days_21d": 0, "rainy_days_14d": 2, "tmax_peak_21d": 28.5,
                  "last_date": "2026-10-04"}, "weekly": [], "source": "test"}


def dms(v):
    v = abs(v); d = int(v); m = int((v - d) * 60); s = (v - d - m / 60) * 3600
    return (float(d), float(m), float(s))


def jpeg(lat=None, lon=None, color=(90, 120, 60)) -> bytes:
    im = Image.new("RGB", (256, 256), color)
    exif = Image.Exif()
    if lat is not None:
        exif[0x8825] = {1: "S" if lat < 0 else "N", 2: dms(lat), 3: "W" if lon < 0 else "E", 4: dms(lon)}
        exif[0x0132] = "2026:10:03 10:15:00"
    buf = io.BytesIO(); im.save(buf, "JPEG", exif=exif); return buf.getvalue()


def field(**kw) -> FieldInput:
    base = dict(sowing_date=date.today() - timedelta(days=140), growth_stage="Grain fill", photo_area_m2=0.25,
                polygon=POLY, soil=SoilInput(ph=5.6, organic_matter_pct=1.8, p_level="Low"))
    base.update(kw)
    return FieldInput(**base)


class FakeDetector:
    """Stands in for the GPU detector: returns a different, known count for each photo."""
    def __init__(self, counts):
        self.counts, self.i = list(counts), 0

    def detect(self, image, imgsz=1024, iou=0.7):
        n = self.counts[self.i % len(self.counts)]; self.i += 1
        return wd.Detection(boxes=np.tile(np.array([[0, 0, 10, 10]], np.float32), (n, 1)), scores=np.full(n, 0.9, np.float32),
                            width=image.width, height=image.height, ms=5.0, floor=0.1)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(geo, "reverse_geocode", lambda lat, lon: {"label": "Testville, Nowhere", "country": "Nowhere"})
    monkeypatch.setattr(weather, "get_weather", lambda *a, **k: WX)
    monkeypatch.setattr(soil, "estimate_soilgrids", lambda lat, lon: None)


def run_graph(f: FieldInput, photos, counts=(30, 34, 38, 42, 46, 50)):
    app = graph.build_graph()
    return app.invoke({"field": f, "photos": photos, "status": "running"}, {"configurable": {"detector": FakeDetector(counts)}})


# ----------------------------------------------------------------------------- geometry and EXIF
def test_exif_gps_roundtrip():
    m = geo.read_exif(jpeg(-35.1025, 147.3725))
    assert m["has_gps"] and m["lat"] == pytest.approx(-35.1025, abs=1e-4) and m["lon"] == pytest.approx(147.3725, abs=1e-4)
    assert m["time"] == "2026-10-03T10:15:00"


def test_no_gps_is_reported_as_missing():
    m = geo.read_exif(jpeg())
    assert not m["has_gps"] and m["lat"] is None


def test_distance_area_and_containment():
    assert geo.haversine_m(-35.1, 147.37, -35.101, 147.37) == pytest.approx(111, abs=2)
    assert geo.polygon_area_ha(POLY) == pytest.approx(20.2, abs=0.3)
    assert geo.point_in_polygon(-35.102, 147.372, POLY) and not geo.point_in_polygon(-35.2, 147.4, POLY)
    assert geo.compass(1, 0) == "east" and geo.compass(0, 1) == "north"


# ----------------------------------------------------------------------------- engine
def test_yield_formula_matches_the_documented_example():
    f = field(kernels_per_head=35, tkw_g=40)
    agg = {"density_mean": 500, "ci_low": 500, "ci_high": 500}
    y = engine.yield_range(agg, f)
    assert y["central"] == pytest.approx(7.0, abs=1e-9)           # 500 x 35 x 40 / 100000
    assert y["low"] < y["central"] < y["high"]


def test_calibration_scales_counts_and_is_bounded():
    f = field()
    photos = [{"name": f"p{i}", "usable": True, "count_raw": 40, "hand_count": 48} for i in range(3)]
    a = engine.aggregate(photos, f)
    assert a["calibrated"] and a["calibration_factor"] == pytest.approx(1.2)
    assert photos[0]["count"] == pytest.approx(48)
    wild = [{"name": "p", "usable": True, "count_raw": 10, "hand_count": 100} for _ in range(2)]
    assert engine.aggregate(wild, f)["calibration_factor"] == 1.5     # clamped


def test_ids_are_sequential_and_unique():
    L = engine.Ledger()
    assert [L.fact("a", 1), L.fact("b", 2)] == ["E1", "E2"]
    assert L.driver("x", 1, "t", ["E1"]) == "D1" and L.action("this_season", "t", [], False) == "A1"


def test_verdict_logic_good_watch_poor():
    def verdict(counts, **kw):
        photos = [{"name": f"p{i}", "usable": True, "count_raw": c, "hand_count": None, "lat": None} for i, c in enumerate(counts)]
        f = field(soil=SoilInput(ph=6.5, organic_matter_pct=3.0), **kw)
        wx = {"summary": {**WX["summary"], "water_balance_28d_mm": 0}}
        return engine.analyze(f, photos, {"label": "x", "confidence": "high"}, wx, soil.build_soil(f.soil, None, None), [])["verdict"]
    assert verdict([125] * 12) == "Good"                         # 500 heads/m2 at 0.25 m2
    assert verdict([60] * 12) in ("Poor", "Watch")               # 240 heads/m2, well under the band
    assert verdict([125, 40, 125, 40, 125, 40, 125, 40, 125, 40]) in ("Watch", "Poor")   # very uneven


def test_estimated_soil_is_never_confused_with_a_users_own_test():
    f = field(soil=SoilInput(ph=5.6))
    s = soil.build_soil(f.soil, None, None)
    assert s["field_source"]["ph"] == "soil test"


# ----------------------------------------------------------------------------- validator
def packet():
    f = field()
    photos = [{"name": f"p{i}", "usable": True, "count_raw": 60, "hand_count": None, "lat": None} for i in range(12)]
    return engine.analyze(f, photos, {"label": "Testville", "confidence": "high"}, WX, soil.build_soil(f.soil, None, None), [])["packet"]


def good_draft(p) -> ReportDraft:
    major = next(d for d in p["drivers"] if d["severity"] >= 2)
    must = [a for a in p["candidate_actions"] if a["must_include"]] or p["candidate_actions"][:1]
    return ReportDraft(
        summary="The field is in poor condition and needs attention.",
        reasons=[Reason(text=f"{d['label']} matters here.", evidence_ids=d["evidence"]) for d in p["drivers"] if d["severity"] >= 2 and d["evidence"]],
        actions=[ActionNote(action_id=a["id"], explanation="This follows from the evidence.", evidence_ids=a["evidence"]) for a in must],
        open_questions=[], limitations="Decision support only.")


def test_a_clean_draft_passes():
    p = packet()
    assert llm.validate(good_draft(p), p) == []


def test_validator_blocks_invented_numbers():
    p = packet(); d = good_draft(p); d.summary = "Yield will reach 98765 tonnes per hectare."
    assert any("98765" in e for e in llm.validate(d, p))


def test_validator_blocks_doses_and_products():
    p = packet(); d = good_draft(p)
    d.actions[0].explanation = "Apply tebuconazole at 150 kg/ha."
    errs = llm.validate(d, p)
    assert any("tebuconazole" in e for e in errs) and any("application rate" in e for e in errs)


def test_validator_blocks_unknown_ids_and_raw_ids_in_text():
    p = packet(); d = good_draft(p)
    d.reasons[0].evidence_ids = ["E999"]; d.summary = "See D1 for details."
    errs = llm.validate(d, p)
    assert any("E999" in e for e in errs) and any("raw id" in e for e in errs)


def test_validator_requires_must_include_actions():
    p = packet(); d = good_draft(p); d.actions = []
    assert any("required but missing" in e for e in llm.validate(d, p)) or not any(a["must_include"] for a in p["candidate_actions"])


def test_the_safe_fallback_passes_its_own_validator():
    p = packet()
    assert llm.validate(llm.fallback_draft(p), p) == []


# ----------------------------------------------------------------------------- the LangGraph workflow
def test_full_run_with_gps_photos():
    photos = [PhotoIn(f"p{i}.jpg", jpeg(-35.1005 - 0.0004 * i, 147.3705 + 0.0004 * i)) for i in range(8)]
    out = run_graph(field(), photos)
    r = out["report"]
    assert out["status"] == "done" and r["location"]["how"].startswith("median of GPS")
    assert r["location"]["n_inside"] == 8 and r["location"]["confidence"] == "high"
    assert r["aggregate"]["n_used"] == 8 and r["stats"]["used_fallback"] is True      # LLM disabled in tests
    assert {"intake", "detect_photos", "locate", "weather", "soil", "find_gaps", "analyze", "finalize"} <= set(r["timings"]) | {"finalize"}
    assert all(a["id"].startswith("A") for a in r["actions"]) and r["evidence"]


def test_stops_early_before_heads_are_visible():
    out = run_graph(field(growth_stage="Tillering"), [PhotoIn("a.jpg", jpeg(-35.1, 147.37))])
    assert out["status"] == "needs_input" and "not visible" in out["issues"][0]["message"]


def test_asks_for_a_location_when_there_is_no_gps_and_no_pin():
    out = run_graph(field(), [PhotoIn("a.jpg", jpeg())])
    assert out["status"] == "needs_input" and any("find the field" in i["message"] for i in out["issues"])


def test_a_pin_replaces_missing_gps_and_an_unreadable_file_is_skipped():
    photos = [PhotoIn("a.jpg", jpeg()), PhotoIn("b.jpg", jpeg()), PhotoIn("notes.txt", b"not an image")]
    out = run_graph(field(lat=-35.102, lon=147.372), photos)
    r = out["report"]
    assert out["status"] == "done" and r["location"]["how"] == "pin entered by the user"
    assert r["aggregate"]["n_photos"] == 3 and r["aggregate"]["n_used"] == 2
    assert any("notes.txt" in i["message"] for i in r["issues"])


def test_no_photo_area_means_no_yield_but_still_a_report():
    out = run_graph(field(photo_area_m2=None), [PhotoIn(f"p{i}.jpg", jpeg(-35.101, 147.372)) for i in range(4)])
    r = out["report"]
    assert r["yield"] is None and any("ground area" in g["message"] for g in r["gaps"])


def test_yield_in_tonnes_per_hectare_is_allowed_but_doses_are_not():
    p = packet(); y = next(e for e in p["evidence"] if e["label"].startswith("Yield forecast"))
    d = good_draft(p)
    d.summary = f"The central yield forecast is {y['value']} t/ha."
    assert llm.validate(d, p) == []
    d.summary = "Apply 2 t/ha of lime."                      # 2 is not a packet number, and the sentence is not about yield
    assert any("application rate" in e for e in llm.validate(d, p))
    d.summary = f"Apply {y['value']} kg/ha of nitrogen."     # kg/ha is always a dose, even with a packet number
    assert any("application rate" in e for e in llm.validate(d, p))
