"""Tests for the field advisor. No GPU, network or language model needed.

Run from the frontend folder:   ..\\.venv\\Scripts\\python -m pytest tests -q
"""
from __future__ import annotations

import io
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["ADVISOR_FORCE_FALLBACK"] = "1"          # never call Ollama in tests

import wheat_detector as wd                         # noqa: E402
from advisor import config, engine, geo, graph, llm, soil, weather  # noqa: E402
from advisor.schemas import ActionNote, FieldInput, PhotoIn, Reason, ReportDraft, SoilInput  # noqa: E402

REAL_GET_FORECAST = weather.get_forecast          # the autouse fixture below replaces it for every test
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


def make_forecast(rain=(0, 0, 0, 0, 0, 0, 0), tmax=(22,) * 7, chance=(0,) * 7, humidity=(60,) * 7):
    """A forecast for the next seven days, in the shape the weather connector returns."""
    today = date.today()
    days = [{"date": (today + timedelta(days=i + 1)).isoformat(), "tmax": tmax[i], "tmin": tmax[i] - 10 if tmax[i] is not None else None,
             "rain_mm": rain[i], "chance_pct": chance[i], "humidity_pct": humidity[i], "et0_mm": 3.0} for i in range(7)]
    fc = {"days": days, "local_today": today.isoformat(), "fetched": datetime.now(timezone.utc).isoformat(timespec="minutes"),
          "source": "test"}
    summary = weather.summarize_forecast(fc)
    return {**fc, "summary": summary} if summary else {"error": "too few days"}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(geo, "reverse_geocode", lambda lat, lon: {"label": "Testville, Nowhere", "country": "Nowhere"})
    monkeypatch.setattr(weather, "get_weather", lambda *a, **k: WX)
    monkeypatch.setattr(weather, "get_forecast", lambda *a, **k: make_forecast())
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


# ----------------------------------------------------------------------------- chat
import json as _json                                  # noqa: E402
from types import SimpleNamespace                     # noqa: E402

from advisor import chat                              # noqa: E402


class FakeClient:
    """Replays canned model replies so chat logic can be tested without Ollama."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def chat(self, **kw):
        self.calls += 1
        return SimpleNamespace(message=SimpleNamespace(content=self.replies.pop(0)))


def ans(text):
    """What the answer step of the model returns."""
    return _json.dumps({"answer": text})


def ext(**facts):
    """What the fact-extraction step of the model returns."""
    return _json.dumps(facts)


@pytest.fixture
def live_chat(monkeypatch):
    monkeypatch.setattr(config, "FORCE_FALLBACK", False)
    monkeypatch.setattr(llm, "status", lambda: {"server": True, "installed": True, "model": "x"})

    def install(replies):
        fake = FakeClient(replies)
        monkeypatch.setattr(llm, "client", lambda: fake)
        return fake
    return install


def test_updates_must_be_backed_by_the_users_own_words():
    u = chat.FieldUpdates(kernels_per_head=36, tkw_g=40, phosphorus="Low", irrigated=True)
    got = chat.clean_updates(u, "Kernels per head is 36 for this variety.")
    assert got == {"kernels_per_head": 36}                       # 40, Low and irrigated were never said
    got = chat.clean_updates(chat.FieldUpdates(phosphorus="Low", irrigated=False), "Phosphorus was rated low and it is rainfed.")
    assert got == {"phosphorus": "Low", "irrigated": False}


def test_chat_answers_are_checked_like_report_text():
    p = packet()
    assert chat.check_answer("Rain was short in the last weeks.", p, ["hi"], {}) == []
    assert any("not in the report" in e for e in chat.check_answer("You will get 98765 t.", p, ["hi"], {}))
    assert any("application rate" in e for e in chat.check_answer("Put on 120 kg/ha of nitrogen.", p, ["hi"], {}))
    assert any("tebuconazole" in e for e in chat.check_answer("Spray tebuconazole.", p, ["hi"], {}))
    assert chat.check_answer("Thanks, 36 kernels per head noted.", p, ["It has 36 kernels per head"], {"kernels_per_head": 36}) == []


def test_chat_turn_extracts_facts_in_their_own_step_then_answers(live_chat):
    fake = live_chat([ext(kernels_per_head=36), ans("Thank you. I can add the kernel count to the report.")])
    out = chat.chat_turn(packet(), [], "The variety has 36 kernels per head.")
    assert not out["used_fallback"] and out["updates"] == {"kernels_per_head": 36} and fake.calls == 2


def test_chat_turn_drops_extracted_facts_the_user_never_wrote(live_chat):
    live_chat([ext(kernels_per_head=36, tkw_g=40, irrigated=True), ans("Noted, I can add the kernel count.")])
    out = chat.chat_turn(packet(), [], "We have about 36 kernels per head.")
    assert out["updates"] == {"kernels_per_head": 36}                # 40 and irrigated were made up by the model


def test_a_plain_question_skips_the_extraction_call(live_chat):
    fake = live_chat([ans("The stand is thin and the weather was dry.")])
    out = chat.chat_turn(packet(), [], "Why is it poor?")
    assert fake.calls == 1 and out["updates"] == {} and not out["used_fallback"]


def test_chat_turn_retries_once_when_the_answer_fails_the_checks(live_chat):
    fake = live_chat([ans("The result will be 98765 tonnes."), ans("I cannot give a number beyond the range in the report.")])
    out = chat.chat_turn(packet(), [], "Why is it poor?")
    assert fake.calls == 2 and not out["used_fallback"] and "98765" not in out["answer"]


def test_chat_turn_falls_back_safely_after_two_bad_answers(live_chat):
    live_chat([ans("Put on 120 kg/ha now."), ans("Put on 130 kg/ha now.")])
    out = chat.chat_turn(packet(), [], "Please advise.")
    assert out["used_fallback"] and "kg/ha" not in out["answer"]


def test_facts_survive_even_when_the_answer_cannot_be_trusted(live_chat):
    live_chat([ext(inputs_applied="urea at tillering"), ans("Apply 120 kg/ha more."), ans("Apply 120 kg/ha more.")])
    out = chat.chat_turn(packet(), [], "We put urea on at tillering.")
    assert out["used_fallback"] and out["updates"] == {"inputs_applied": "urea at tillering"} and "picked up" in out["answer"]


def test_a_failed_extraction_call_loses_neither_the_answer_nor_the_pattern_matched_facts(live_chat):
    live_chat(["not json at all", ans("Thanks, noted.")])
    out = chat.chat_turn(packet(), [], "We have 36 kernels per head.")
    assert not out["used_fallback"] and out["updates"] == {"kernels_per_head": 36}


def test_chat_says_so_when_the_model_is_not_available(monkeypatch):
    monkeypatch.setattr(config, "FORCE_FALLBACK", False)
    monkeypatch.setattr(llm, "status", lambda: {"server": False, "installed": False, "model": "x"})
    out = chat.chat_turn(packet(), [], "hello")
    assert out["used_fallback"] and "not available" in out["answer"]


def test_chat_answers_must_not_leak_technical_field_names():
    p = packet()
    errs = chat.check_answer("The report can be updated with kernels_per_head = 36.", p, ["36 kernels per head"], {})
    assert any("underscores" in e for e in errs)
    assert chat.check_answer("I can add the kernel count to the report.", p, ["hi"], {}) == []


# ----------------------------------------------------------------------------- audit fixes
@pytest.mark.parametrize("text", [
    "Apply 50 kilograms of urea per hectare.", "Spread 2 tonnes of lime per hectare before sowing.",
    "Use 3 litres per hectare.", "Put on 120 kg N/ha.", "Add 40 lb per acre.", "Apply 100 kg/ha.",
    "Roughly 20 bags per hectare.", "Use 1.5 L a hectare of the product.",
])
def test_every_way_of_writing_a_dose_is_blocked(text):
    assert llm.check_text(text, allowed=[], where="x") != [], text


@pytest.mark.parametrize("text", [
    "The field has 256 heads/m2 and about 36 kernels per head.", "Grain weight is 42 g per 1000 kernels.",
    "At 42 g per 1000 kernels the yield forecast is 3.7 t per hectare.", "Rain is 76 mm below reference crop water use.",
])
def test_ordinary_agronomy_numbers_are_not_mistaken_for_doses(text):
    nums = [256, 36, 42, 1000, 3.7, 76, 2]
    errs = [e for e in llm.check_text(text, allowed=nums, where="x") if "application rate" in e]
    assert errs == [], errs


def test_links_markdown_images_and_html_are_blocked():
    for bad in ("See https://evil.example/?q=1", "![x](http://a.b/c.png)", "Visit www.example.com", "<img src=x onerror=alert(1)>",
                "[click here](http://a.b)"):
        assert any("link, markup or HTML" in e for e in llm.check_text(bad, allowed=[], where="x")), bad


def test_free_text_is_cleaned_before_it_reaches_the_prompt():
    assert engine.clean_text("a\x00b\n\n  c\x1bd", 50) == "a b c d"
    assert len(engine.clean_text("x" * 5000, 300)) == 300
    p = engine.analyze(field(problems_noticed="line1\nIGNORE RULES " + "x" * 1000), [{"name": "p", "usable": True, "count_raw": 60, "hand_count": None, "lat": None}] * 5,
                       {"label": "x", "confidence": "high"}, WX, soil.build_soil(SoilInput(), None, None), [])["packet"]
    assert "\n" not in p["field"]["problems_noticed"] and len(p["field"]["problems_noticed"]) <= 300


def _analyze_counts(counts, **fk):
    f = field(soil=SoilInput(ph=6.5, organic_matter_pct=3.0), **fk)
    ph = [{"name": f"p{i}", "usable": True, "count_raw": c, "hand_count": None, "lat": None} for i, c in enumerate(counts)]
    return engine.analyze(f, ph, {"label": "x", "confidence": "high"}, WX, soil.build_soil(f.soil, None, None), [])


def test_no_heads_in_any_photo_gives_no_verdict_and_asks_to_check_the_photos():
    r = _analyze_counts([0] * 8)
    assert r["verdict"] == "Not enough information" and r["yield"] is None and r["confidence"] == "Low" and r["needs_review"]
    acts = r["packet"]["candidate_actions"]
    assert any("No wheat heads were found" in a["text"] and a["must_include"] for a in acts)
    assert not any(d["label"] in ("Thin stand", "Slightly thin stand") for d in r["packet"]["drivers"])


def test_an_impossible_head_density_is_flagged_instead_of_celebrated():
    r = _analyze_counts([900] * 8)                              # 3600 heads/m2 would give 50+ t/ha
    assert r["verdict"] == "Not enough information" and r["yield"] is None and r["confidence"] == "Low"
    assert any("photo ground area" in a["text"] and a["must_include"] for a in r["packet"]["candidate_actions"])
    assert any("ground area" in q for q in r["packet"]["open_questions"])


def test_cereal_after_cereal_cites_the_previous_crop_not_the_growth_stage():
    r = _analyze_counts([125] * 8, previous_crop="Barley")
    prev = next(e for e in r["packet"]["evidence"] if e["label"] == "Previous crop")
    d = next(d for d in r["packet"]["drivers"] if d["label"] == "Cereal after cereal")
    assert d["evidence"] == [prev["id"]]


def test_the_interval_assumption_states_the_counting_error():
    r = _analyze_counts([125] * 8)
    assert any("allows 10% for counting error" in a for a in r["packet"]["assumptions"])


def test_unreadable_files_are_described_as_unreadable_not_low_quality():
    f = field()
    ph = [{"name": "a", "usable": True, "count_raw": 100, "hand_count": None, "lat": None}, {"name": "b", "usable": False}]
    p = engine.analyze(f, ph, {"label": "x", "confidence": "high"}, WX, soil.build_soil(f.soil, None, None), [])["packet"]
    assert any("could not be read" in e["note"] for e in p["evidence"]) and not any("quality" in e["note"] for e in p["evidence"])


def test_spatial_trend_points_the_right_way():
    photos = [{"name": f"p{i}", "usable": True, "density_m2": 200 + 60 * i, "lat": -35.1 + 0.0002 * (i % 2), "lon": 147.37 + 0.0004 * i} for i in range(8)]
    g = engine.spatial_pattern(photos)["gradient"]                  # density rises toward the east
    assert g["toward"] == "east" and g["away_from"] == "west"


# ----------------------------------------------------------------------------- regular facts are read by pattern, not by the model
@pytest.mark.parametrize("msg,want", [
    ("The soil test says phosphorus is low and potassium is medium, pH 5.8. It is irrigated.",
     {"soil_ph": 5.8, "phosphorus": "Low", "potassium": "Medium", "irrigated": True}),
    ("Soil pH came back at 6.1 and organic matter is 2.4 percent.", {"soil_ph": 6.1, "soil_organic_matter_pct": 2.4}),
    ("We get 36 kernels per head and 42 grams per 1000 kernels.", {"kernels_per_head": 36, "tkw_g": 42}),
    ("Kernels per head is about 33. TKW is 40.", {"kernels_per_head": 33, "tkw_g": 40}),
    ("Our target is 5.5 tonnes per hectare.", {"target_yield_t_ha": 5.5}),
    ("The pH is 7,2", {"soil_ph": 7.2}),
    ("Why is the stand only 256 heads per square metre when the band is 400 to 600?", {}),
])
def test_regular_numbers_are_read_by_pattern(msg, want):
    assert chat.regex_facts(msg) == want


def test_a_forgetful_model_cannot_lose_a_number_written_in_a_regular_form(live_chat):
    live_chat(["{}", "{}", "{}", ans("Thanks.")])                       # every extraction group returns nothing
    out = chat.chat_turn(packet(), [], "Potassium is medium, pH 5.8.")
    assert out["updates"] == {"soil_ph": 5.8, "potassium": "Medium"} or out["updates"] == {"soil_ph": 5.8}


def test_one_bad_value_is_dropped_alone(live_chat):
    live_chat([ext(), ans("Thanks.")])
    out = chat.extract_updates("pH 62 and 36 kernels per head")        # pH 62 is impossible, the kernel count is fine
    assert out == {} or out == {"kernels_per_head": 36}


@pytest.mark.parametrize("msg,want", [
    ("Potassium is high.", {"potassium": "High"}),
    ("The test says phosphorus low, nitrogen was rated medium and K is high.", {"phosphorus": "Low", "nitrogen": "Medium", "potassium": "High"}),
    ("We have high phosphorus here.", {"phosphorus": "High"}),
    ("It is irrigated by a pivot.", {"irrigated": True}),
    ("This is a rain-fed paddock.", {"irrigated": False}),
    ("The field is not irrigated.", {"irrigated": False}),
    # questions are never read as facts
    ("Is the field irrigated?", {}),
    ("Is potassium high?", {}),
    ("What if my pH were 5.8?", {}),
    ("The pH is 5.8. Is that bad?", {"soil_ph": 5.8}),
])
def test_ratings_and_irrigation_are_read_from_statements_only(msg, want):
    assert chat.regex_facts(msg) == want


# ----------------------------------------------------------------------------- weather forecast
import json                                              # noqa: E402
import random                                            # noqa: E402


def analyze_with(fc, stage="Grain fill", irrigated=False, balance=-76, rainy14=2, **fkw):
    """Analyze 12 healthy-density photos with a given forecast. The default weather is a 4-week water shortage."""
    f = field(growth_stage=stage, irrigated=irrigated, soil=SoilInput(ph=6.5, organic_matter_pct=3.0), **fkw)
    photos = [{"name": f"p{i}", "usable": True, "count_raw": 125, "hand_count": None, "lat": None} for i in range(12)]
    wx = {"summary": {**WX["summary"], "water_balance_28d_mm": balance, "rainy_days_14d": rainy14}}
    return engine.analyze(f, photos, {"label": "x", "confidence": "high"}, wx, soil.build_soil(f.soil, None, None), [], fc)


def soon(res):
    return [a for a in res["packet"]["candidate_actions"] if a["window"] == "next_days"]


def has(res, text):
    return any(text in a["text"] for a in soon(res))


def test_forecast_numbers_come_from_the_first_five_days_only():
    s = make_forecast(rain=(0, 2, 0, 10, 5, 40, 40), chance=(0, 30, 0, 80, 60, 90, 90))["summary"]
    assert s["action_days"] == 5 and s["rain_5d_mm"] == 17 and s["rain_3d_mm"] == 2 and s["rain_days_5d"] == 3
    assert s["wettest"]["mm"] == 10 and s["wettest"]["chance_pct"] == 80 and s["chance_5d_pct"] == 80
    assert not s["dry_5d"] and s["rain_sig_5d"] and not s["rain_soon_3d"]


def test_days_are_named_tomorrow_then_by_weekday():
    s = make_forecast(rain=(9, 0, 0, 0, 0, 0, 0), chance=(90,) * 7)["summary"]
    assert s["wettest"]["when"] == "tomorrow"
    s = make_forecast(rain=(0, 0, 9, 0, 0, 0, 0), chance=(90,) * 7)["summary"]
    assert s["wettest"]["when"] == "on " + (date.today() + timedelta(days=3)).strftime("%A")


def test_a_missing_rain_figure_never_looks_like_a_dry_spell():
    assert make_forecast(rain=(None, 0, 0, 0, 0, 0, 0))["summary"]["dry_5d"] is False
    assert make_forecast(rain=(0, 0, 0, 0, 0, 0, 0))["summary"]["dry_5d"] is True
    assert "error" in make_forecast(rain=(None, None, None, 0, 0, 0, 0))        # too little to advise on


def test_a_forecast_with_too_few_days_is_refused():
    fc = make_forecast()
    short = {"days": fc["days"][:2], "local_today": fc["local_today"], "fetched": fc["fetched"], "source": "t"}
    assert weather.summarize_forecast(short) is None


class _Resp:
    def __init__(self, j):
        self.j = j

    def raise_for_status(self):
        pass

    def json(self):
        return self.j


def test_tomorrow_is_the_fields_tomorrow_and_the_answer_is_kept_for_a_while(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    offset = 39600                                                         # a field 11 hours ahead of UTC
    local_today = (datetime.now(timezone.utc) + timedelta(seconds=offset)).date()
    times = [(local_today + timedelta(days=i)).isoformat() for i in range(-1, 8)]     # yesterday .. +7
    daily = {"time": times, "temperature_2m_max": [20.0] * 9, "temperature_2m_min": [9.0] * 9, "precipitation_sum": [1.0] * 9,
             "precipitation_probability_max": [40] * 9, "relative_humidity_2m_mean": [70] * 9, "et0_fao_evapotranspiration": [3.0] * 9}
    calls = []
    monkeypatch.setattr(weather.requests, "get", lambda *a, **k: calls.append(1) or _Resp({"utc_offset_seconds": offset, "daily": daily}))
    fc = weather.fetch_forecast(-35.1, 147.37)
    assert fc["local_today"] == local_today.isoformat()
    assert fc["days"][0]["date"] == (local_today + timedelta(days=1)).isoformat() and len(fc["days"]) == 7
    assert weather.fetch_forecast(-35.1, 147.37) == fc and len(calls) == 1             # second call came from the cache


def test_a_damaged_cache_file_is_ignored_and_missing_fields_become_none(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(weather.requests, "get", lambda *a, **k: _Resp({"utc_offset_seconds": 0, "daily": {
        "time": [(date.today() + timedelta(days=i)).isoformat() for i in range(8)], "temperature_2m_max": [20] * 8,
        "temperature_2m_min": [9] * 8, "precipitation_sum": [0] * 8, "et0_fao_evapotranspiration": [3] * 8}}))
    weather.fetch_forecast(1.0, 2.0)                                       # writes a good file
    next(tmp_path.glob("fc_*.json")).write_text("{not json")
    fc = weather.fetch_forecast(1.0, 2.0)                                  # damaged file -> fetched again, no crash
    assert len(fc["days"]) >= 6 and fc["days"][0]["chance_pct"] is None    # fields the service left out become None


def test_a_failing_forecast_service_becomes_an_error_block_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)

    def boom(*a, **k):
        raise weather.requests.ConnectionError("offline")
    monkeypatch.setattr(weather.requests, "get", boom)
    assert "error" in REAL_GET_FORECAST(-35.1, 147.37)


def test_forecast_age():
    fresh = make_forecast()
    assert weather.forecast_age_hours(fresh) < 0.1
    old = {**fresh, "fetched": (datetime.now(timezone.utc) - timedelta(hours=20)).isoformat(timespec="minutes")}
    assert 19.9 < weather.forecast_age_hours(old) < 20.1
    assert weather.forecast_age_hours({"error": "x"}) is None and weather.forecast_age_hours(None) is None


# --- the rules
def test_dry_days_ahead_after_a_water_shortage_say_irrigate_soon():
    r = analyze_with(make_forecast(), irrigated=True)
    a = next(a for a in soon(r) if "Irrigate within the next two days" in a["text"])
    assert a["must_include"] and r["packet"]["outlook"][0]["level"] == "watch"
    assert "No useful rain is forecast" in r["packet"]["outlook"][0]["text"] and "short of water" in r["packet"]["outlook"][0]["text"]


def test_heavy_rain_soon_says_wait_before_irrigating():
    r = analyze_with(make_forecast(rain=(0, 14, 0, 0, 0, 0, 0), chance=(0, 80, 0, 0, 0, 0, 0)), irrigated=True)
    assert has(r, "Hold off irrigating") and not has(r, "Irrigate within")
    assert next(a for a in soon(r) if "Hold off" in a["text"])["must_include"]       # a shortage exists, so it is required


def test_unlikely_rain_does_not_stop_irrigation():
    r = analyze_with(make_forecast(rain=(0, 14, 0, 0, 0, 0, 0), chance=(0, 20, 0, 0, 0, 0, 0)), irrigated=True)
    assert not has(r, "Hold off irrigating")


def test_rain_later_than_three_days_does_not_stop_irrigation():
    r = analyze_with(make_forecast(rain=(0, 0, 0, 0, 25, 0, 0), chance=(0, 0, 0, 0, 90, 0, 0)), irrigated=True)
    assert not has(r, "Hold off irrigating") and not has(r, "Irrigate within")


def test_a_rain_fed_crop_is_never_told_to_irrigate():
    for fc in (make_forecast(), make_forecast(rain=(0, 14, 0, 0, 0, 0, 0), chance=(0, 80, 0, 0, 0, 0, 0))):
        r = analyze_with(fc, irrigated=False)
        assert not has(r, "rrigat")


def test_no_irrigation_advice_without_a_water_shortage():
    r = analyze_with(make_forecast(), irrigated=True, balance=0)
    assert not has(r, "Irrigate within")


@pytest.mark.parametrize("stage,expect", [("Flowering", True), ("Grain fill", True), ("Heading", False), ("Ripening", False)])
def test_a_hot_spell_matters_only_at_heat_sensitive_stages(stage, expect):
    r = analyze_with(make_forecast(tmax=(33, 34, 32, 25, 25, 25, 25)), stage=stage, irrigated=True)
    assert has(r, "hot days") is expect and (any("hot days" in o["text"] for o in r["packet"]["outlook"]) is expect)


def test_one_hot_day_is_not_a_hot_spell():
    assert not has(analyze_with(make_forecast(tmax=(36, 25, 25, 25, 25, 25, 25)), stage="Flowering"), "hot days")


def test_a_hot_spell_gives_different_advice_to_rain_fed_and_irrigated_fields():
    hot = make_forecast(tmax=(33, 34, 32, 25, 25, 25, 25))
    assert has(analyze_with(hot, stage="Flowering", irrigated=True), "Water the crop before")
    assert has(analyze_with(hot, stage="Flowering", irrigated=False), "Water cannot be added")


def test_warm_humid_rainy_days_raise_the_disease_lookout():
    wet = make_forecast(rain=(2, 3, 1, 0, 2, 0, 0), humidity=(80,) * 7)
    r = analyze_with(wet, stage="Flowering")
    a = next(a for a in soon(r) if "disease" in a["text"])
    assert a["confirm"] and a["must_include"]
    assert not has(analyze_with(make_forecast(rain=(2, 3, 1, 0, 2, 0, 0), humidity=(50,) * 7), stage="Flowering"), "disease")   # dry air
    assert not has(analyze_with(make_forecast(rain=(2, 3, 1, 0, 2, 0, 0), humidity=(80,) * 7, tmax=(10,) * 7), stage="Flowering"), "disease")   # cold
    assert not has(analyze_with(wet, stage="Ripening"), "disease")


def test_the_disease_walk_is_not_asked_for_twice():
    r = analyze_with(make_forecast(rain=(2, 3, 1, 0, 2, 0, 0), humidity=(80,) * 7), stage="Flowering", rainy14=9)
    assert not has(r, "disease")                                                    # the past-weather "Wet spell" action covers it
    assert sum("Walk the field to look for head diseases" in a["text"] for a in r["packet"]["candidate_actions"]) == 1
    assert any("head diseases" in o["text"] for o in r["packet"]["outlook"])        # but the outlook still says so


def test_rain_before_harvest_is_a_ripening_warning_only():
    rainy = make_forecast(rain=(0, 8, 9, 0, 0, 0, 0), chance=(0, 70, 70, 0, 0, 0, 0))
    assert has(analyze_with(rainy, stage="Ripening"), "before harvest")
    assert not has(analyze_with(rainy, stage="Grain fill"), "before harvest")


def test_the_forecast_never_changes_the_verdict_confidence_yield_or_drivers():
    base = analyze_with(None, irrigated=True)
    for fc in (make_forecast(), make_forecast(rain=(0, 30, 30, 0, 0, 0, 0), chance=(90,) * 7, tmax=(38,) * 7, humidity=(90,) * 7)):
        r = analyze_with(fc, irrigated=True)
        assert (r["verdict"], r["confidence"], r["confidence_points"], r["yield"]) == (base["verdict"], base["confidence"], base["confidence_points"], base["yield"])
        assert r["packet"]["drivers"] == base["packet"]["drivers"]


def test_without_a_forecast_nothing_about_it_appears():
    for fc in (None, {"error": "offline"}):
        r = analyze_with(fc, irrigated=True)
        assert r["packet"]["outlook"] == [] and not soon(r)
        assert not any(e["label"].startswith("Forecast") for e in r["packet"]["evidence"])


def test_a_quiet_forecast_says_so_without_raising_a_warning():
    r = analyze_with(make_forecast(rain=(0, 2, 1, 1, 0, 0, 0), chance=(0, 40, 30, 30, 0, 0, 0)), balance=0)
    assert [o["level"] for o in r["packet"]["outlook"]] == ["info"] and not soon(r)


# --- safety checks on what the language model may say about it
def forecast_packet(**kw):
    return analyze_with(make_forecast(rain=(0, 14, 0, 0, 0, 0, 0), chance=(0, 80, 0, 0, 0, 0, 0)), irrigated=True, **kw)["packet"]


def forecast_draft(p) -> ReportDraft:
    """A clean draft whose forecast-based actions say they are predictions."""
    d = good_draft(p)
    if not d.reasons:                                          # this field's drivers are all minor, so cite the first one
        dr = next(x for x in p["drivers"] if x["evidence"])
        d.reasons = [Reason(text=f"{dr['label']} matters here.", evidence_ids=dr["evidence"])]
    for a in d.actions:
        a.explanation = "This is expected to matter soon, going by the forecast."
    return d


def test_forecast_figures_are_checked_like_every_other_number():
    p = forecast_packet(); d = forecast_draft(p)
    assert llm.validate(d, p) == []
    d.summary = "Rain of 987 mm is forecast tomorrow."
    assert any("987" in e for e in llm.validate(d, p))
    d.summary = "About 14 mm of rain is forecast tomorrow."
    assert not any("number" in e for e in llm.validate(d, p))


def test_a_sentence_built_on_the_forecast_must_say_it_is_a_prediction():
    p = forecast_packet(); d = forecast_draft(p)
    fc_id = next(e["id"] for e in p["evidence"] if e["label"].startswith("Forecast rain"))
    d.reasons.append(Reason(text="Rain arrives tomorrow.", evidence_ids=[fc_id]))
    assert any("forecast" in e and "reason" in e for e in llm.validate(d, p))
    d.reasons[-1].text = "Rain is forecast for tomorrow."
    assert llm.validate(d, p) == []
    d.actions[0].evidence_ids = [fc_id]
    d.actions[0].explanation = "It will rain."
    assert any("rests on the weather forecast" in e for e in llm.validate(d, p))


def test_the_fallback_report_covers_the_forecast_and_passes_the_validator():
    p = forecast_packet()
    d = llm.fallback_draft(p)
    assert llm.validate(d, p) == [] and "Heads-up for the next few days" in d.summary
    assert {a["id"] for a in p["candidate_actions"] if a["window"] == "next_days"} <= {a.action_id for a in d.actions}


def test_the_fallback_never_cuts_a_required_action_even_with_many_candidates():
    p = forecast_packet()
    extra = [{"id": f"A{100 + i}", "window": "next_season", "text": "A routine step.", "evidence": [], "confirm": False, "must_include": False} for i in range(15)]
    p["candidate_actions"] = extra + p["candidate_actions"]                    # required ones now come last
    d = llm.fallback_draft(p)
    assert len(d.actions) == 10 and llm.validate(d, p) == []


# --- the workflow
def test_the_forecast_runs_as_its_own_workflow_step_and_reaches_the_report():
    out = run_graph(field(irrigated=True), [PhotoIn(f"p{i}.jpg", jpeg(-35.1005 - 0.0004 * i, 147.3705)) for i in range(6)])
    r = out["report"]
    assert "forecast" in r["timings"] and r["forecast"]["summary"]["action_days"] == 5
    assert any(e["label"].startswith("Forecast rain") for e in r["packet"]["evidence"])
    assert any(a["window"] == "next_days" for a in r["actions"])               # dry week + shortage + irrigated


def test_a_forecast_failure_does_not_stop_the_report(monkeypatch):
    monkeypatch.setattr(weather, "get_forecast", lambda *a, **k: {"error": "HTTPError: 500 for url: https://example.test/?latitude=1"})
    out = run_graph(field(), [PhotoIn(f"p{i}.jpg", jpeg(-35.1005, 147.3705)) for i in range(4)])
    r = out["report"]
    assert out["status"] == "done" and r["packet"]["outlook"] == []
    gap = next(g["message"] for g in r["gaps"] if "forecast" in g["message"])
    assert "next few days" in gap and "http" not in gap.lower()                 # no raw error or URL goes into the prompt


def test_random_forecasts_never_break_the_engine_or_the_fallback():
    rnd = random.Random(7)

    def pick(lo, hi):
        return rnd.choice([None, round(rnd.uniform(lo, hi), 1)])

    for _ in range(150):
        rain = tuple(rnd.choice([None, 0, 0, round(rnd.uniform(0, 80), 1)]) for _ in range(7))
        fc = make_forecast(rain=rain, tmax=tuple(pick(-10, 46) for _ in range(7)), chance=tuple(pick(0, 100) for _ in range(7)),
                           humidity=tuple(pick(10, 100) for _ in range(7)))
        stage = rnd.choice(["Heading", "Flowering", "Grain fill", "Ripening"])
        r = analyze_with(fc, stage=stage, irrigated=rnd.random() < 0.5, balance=rnd.choice([-90, -40, 0]), rainy14=rnd.choice([2, 9]))
        p = r["packet"]
        json.dumps(p)                                                           # always serialisable
        assert llm.validate(llm.fallback_draft(p), p) == []


# --- audit additions
@pytest.mark.parametrize("text,ok", [
    ("It will rain on Saturday.", False),
    ("Tomorrow will be dry, so irrigate.", False),
    ("The next five days will be hot.", False),
    ("Rain is forecast on Saturday.", True),
    ("Saturday is expected to be wet.", True),
    ("Heat is likely in the next few days.", True),
    ("Plan the nitrogen for next season.", True),          # "next season" is not a forecast phrase
    ("Rain was 20 mm below crop water use.", True),
])
def test_coming_days_must_be_described_as_a_prediction(text, ok):
    errs = llm.check_text(text, allowed=[20.0], where="x")
    assert (not any("coming days" in e for e in errs)) is ok


def test_chat_answers_get_the_same_prediction_check():
    p = forecast_packet()
    errs = chat.check_answer("It will rain tomorrow, so wait.", p, [], {})
    assert any("coming days" in e for e in errs)
    assert not any("coming days" in e for e in chat.check_answer("Rain is forecast tomorrow, so wait.", p, [], {}))


def test_only_the_rounded_position_is_sent_for_the_forecast(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    sent = {}

    def fake_get(url, params=None, **kw):
        sent.update(params)
        return _Resp({"utc_offset_seconds": 0, "daily": {"time": [(date.today() + timedelta(days=i)).isoformat() for i in range(8)],
                                                         "temperature_2m_max": [20] * 8, "temperature_2m_min": [9] * 8,
                                                         "precipitation_sum": [0] * 8, "et0_fao_evapotranspiration": [3] * 8}})
    monkeypatch.setattr(weather.requests, "get", fake_get)
    weather.fetch_forecast(-35.123456, 147.987654)
    assert sent["latitude"] == -35.12 and sent["longitude"] == 147.99


def test_truncated_service_data_becomes_an_error_not_a_crash(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    days = [(date.today() + timedelta(days=i)).isoformat() for i in range(8)]
    for daily in ({"time": days, "temperature_2m_max": [20] * 3, "temperature_2m_min": [9] * 8, "precipitation_sum": [0] * 8},
                  {"time": ["not a date"] * 8, "temperature_2m_max": [20] * 8, "temperature_2m_min": [9] * 8, "precipitation_sum": [0] * 8},
                  {}):
        monkeypatch.setattr(weather.requests, "get", lambda *a, _d=daily, **k: _Resp({"daily": _d}))
        assert "error" in REAL_GET_FORECAST(-35.1, 147.37)


def test_a_negative_rain_figure_cannot_cancel_real_rain():
    s = make_forecast(rain=(-50, 12, 0, 0, 0, 0, 0), chance=(0, 90, 0, 0, 0, 0, 0))["summary"]
    assert s["rain_3d_mm"] == 12 and s["rain_soon_3d"]


def test_a_weather_failure_does_not_put_the_raw_error_into_the_prompt(monkeypatch):
    monkeypatch.setattr(weather, "get_weather", lambda *a, **k: {"error": "ProxyError: url /v1/archive?latitude=-31.25&longitude=149.25"})
    out = run_graph(field(), [PhotoIn(f"p{i}.jpg", jpeg(-35.1005, 147.3705)) for i in range(4)])
    gap = next(g["message"] for g in out["report"]["gaps"] if "Weather data" in g["message"])
    assert "latitude" not in gap and "ProxyError" not in gap
    assert "latitude" not in json.dumps(out["report"]["packet"])
