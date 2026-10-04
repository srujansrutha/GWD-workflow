"""Field report page: photos + field details in, a grounded condition report out.

The work happens in a LangGraph workflow (advisor/graph.py). This page collects the inputs, runs the graph while
showing each step, then presents the result.
"""
from __future__ import annotations

import hashlib
import html
import json
from datetime import date, timedelta

import folium
import pandas as pd
import streamlit as st
from folium.plugins import Draw
from pydantic import ValidationError
from streamlit_folium import st_folium

from advisor import config as acfg
from advisor import chat, geo, graph, llm
from advisor import weather as wxmod
from advisor.schemas import LEVELS, TEXTURES, FieldInput, PhotoIn, SoilInput
from common import inject_report_css, load_detector

inject_report_css()

STAGE_DEFAULT = acfg.STAGES.index("Grain fill")
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


def esc(s) -> str:
    return html.escape(str(s), quote=True)


@st.cache_data(show_spinner=False, max_entries=300)
def photo_meta(digest: str, _raw: bytes) -> dict:
    return geo.read_exif(_raw)


@st.cache_data(ttl=5, show_spinner=False)
def llm_status() -> dict:
    return llm.status()


def clear_location() -> None:
    st.session_state.update(pin=None, polygon=[], last_click=None, pin_lat_in=0.0, pin_lon_in=0.0)


FORM_KEYS = {"kernels_per_head": "a_k", "tkw_g": "a_t", "previous_crop": "f_prev", "problems_noticed": "f_problems",
             "irrigated": "f_irr", "target_yield_t_ha": "f_target", "soil_ph": "s_ph", "soil_organic_matter_pct": "s_om",
             "phosphorus": "s_p", "potassium": "s_k", "nitrogen": "s_n"}


def apply_updates(updates: dict, idx: int) -> None:
    """Put what the user said in the chat into the form, then ask for the report to be re-run."""
    for k, v in updates.items():
        if k == "inputs_applied":
            old = st.session_state.get("f_applied", "").strip()
            st.session_state["f_applied"] = (old + "\n" + v).strip() if old else v
        else:
            st.session_state[FORM_KEYS[k]] = v
    st.session_state["fr_chat"][idx]["applied"] = True
    st.session_state["fr_rerun"] = True


def ask_in_chat(question: str) -> None:
    msgs = st.session_state.setdefault("fr_chat", [])
    if not msgs or msgs[-1]["content"] != question:
        msgs.append({"role": "assistant", "content": question})


def step(n: int, title: str, sub: str = "") -> None:
    st.markdown(f'<div class="step-h"><span class="step-n">{n}</span><span class="step-t">{esc(title)}</span>'
                f'<span class="step-s">{esc(sub)}</span></div>', unsafe_allow_html=True)


def ev_chips(ids: list[str], evidence: dict) -> str:
    out = []
    for i in ids:
        e = evidence.get(i)
        tip = f"{e['label']}: {e['value']} {e['unit']}".strip() if e else ""
        out.append(f'<span class="ev" title="{esc(tip)}">{esc(i)}</span>')
    return "".join(out)


# --------------------------------------------------------------------------- header
detector = load_detector()
ls = llm_status()
llm_ok = ls["server"] and ls["installed"]
dot_llm = "dot" if llm_ok else "dot bad"
llm_text = (f"Local model {esc(ls['model'])}" if llm_ok else
            f"Model {esc(ls['model'])} not ready" + ("" if ls["server"] else " (Ollama is not running)"))
st.markdown(
    f"""
<div class="hero">
  <h1>📋 Field report</h1>
  <p>Upload field photos, describe the field, and get a condition report with the reasons behind it and what to do
  about it. Your photos and the AI model stay on this machine. Only the field's approximate position is sent to free
  online services to look up the weather and forecast, the place name and the soil.</p>
  <span class="chip"><span class="{'dot' if detector.on_gpu else 'dot warn'}"></span>{esc(detector.device_name)}</span>
  <span class="chip"><span class="{dot_llm}"></span>{llm_text}</span>
  <span class="chip">Workflow: LangGraph</span>
</div>
""",
    unsafe_allow_html=True,
)
if not llm_ok:
    st.warning("The language model is not available, so the report text will use the plain rule-based wording. "
               "The counts, weather, soil checks and verdict still work.")

# --------------------------------------------------------------------------- 1 field + crop
with st.container(border=True):
    step(1, "Field and crop", "wheat only for now")
    c1, c2, c3 = st.columns(3)
    name = c1.text_input("Field name", "My field", key="f_name")
    variety = c2.text_input("Variety (optional)", "", key="f_variety")
    sowing = c3.date_input("Sowing date", value=date.today() - timedelta(days=140), max_value=date.today(), key="f_sow")
    c4, c5, c6 = st.columns(3)
    stage = c4.selectbox("Growth stage now", acfg.STAGES, index=STAGE_DEFAULT, key="f_stage",
                         help="Heads are only visible from heading onward. Earlier stages cannot be judged from a head count.")
    irrigated = c5.toggle("Irrigated", value=False, key="f_irr")
    target = c6.number_input("Target yield, t/ha (optional)", min_value=0.0, max_value=20.0, value=0.0, step=0.5, key="f_target")
    c7, c8 = st.columns(2)
    previous = c7.text_input("Previous crop", "", key="f_prev", placeholder="for example canola, wheat, fallow")
    problems = c8.text_input("Problems you noticed (optional)", "", key="f_problems")
    applied = st.text_area("Fertilizer and sprays applied so far (optional)", "", key="f_applied", height=70,
                           placeholder="type, rate and date if you have them")

# --------------------------------------------------------------------------- 2 photos
with st.container(border=True):
    step(2, "Photos", "one photo or a whole field of them, any common format")
    uploads = st.file_uploader("Field photos", accept_multiple_files=True, key=f"fr_up_{st.session_state.get('fr_up_key', 0)}",
                               label_visibility="collapsed",
                               help="Use the original files so the GPS stays in them. Messaging apps remove it.")
    cA, cB = st.columns([1.3, 1])
    preset = cA.selectbox("Ground area covered by each photo", ["50 × 50 cm frame (0.25 m²)", "1 m × 1 m frame (1 m²)",
                                                                "Custom", "I don't know"], key="f_area_preset",
                          help="Needed to turn counts into heads per m² and a yield. A quadrat or marker of known size in the photo is the easiest way.")
    custom_area = cB.number_input("Custom area, m²", min_value=0.01, max_value=1000.0, value=0.25, step=0.05,
                                  disabled=preset != "Custom", key="f_area_custom")
    photo_area = {"50 × 50 cm frame (0.25 m²)": 0.25, "1 m × 1 m frame (1 m²)": 1.0, "Custom": custom_area}.get(preset)
    if preset == "I don't know":
        st.caption("Without the photo area the report shows head counts only, with no heads per m² or yield.")

    raws = [(u.name, u.getvalue()) for u in uploads or []]
    digests = [hashlib.md5(r, usedforsecurity=False).hexdigest() for _, r in raws]
    metas = [photo_meta(d, r) for d, (_, r) in zip(digests, raws)]
    hand = [None] * len(raws)
    if raws:
        st.caption(f"{len(raws)} photo{'s' if len(raws) != 1 else ''} uploaded, "
                   f"{sum(1 for m in metas if m['has_gps'])} with GPS. "
                   "Optional: count the heads by hand on two or three photos to correct the counter.")
        df = pd.DataFrame({"Photo": [n for n, _ in raws], "GPS": ["yes" if m["has_gps"] else "no" for m in metas],
                           "Taken": [(m["time"] or "")[:16].replace("T", " ") for m in metas],
                           "Hand count (optional)": pd.array([None] * len(raws), dtype="Int64")})
        ed = st.data_editor(df, hide_index=True, width="stretch", key="hc_" + hashlib.md5("".join(digests).encode(), usedforsecurity=False).hexdigest()[:10],
                            disabled=["Photo", "GPS", "Taken"],
                            column_config={"Hand count (optional)": st.column_config.NumberColumn(min_value=0, max_value=2000, step=1)})
        hand = [None if pd.isna(v) else int(v) for v in ed["Hand count (optional)"]]

# --------------------------------------------------------------------------- 3 location
with st.container(border=True):
    step(3, "Location", "click the map to place the field pin; draw the field boundary with the shape tool")
    gps_pts = [(m["lat"], m["lon"], n) for m, (n, _) in zip(metas, raws) if m["has_gps"]]
    st.session_state.setdefault("pin", None)
    st.session_state.setdefault("polygon", [])
    st.session_state.setdefault("last_click", None)

    lm, rm = st.columns([2.2, 1])
    with lm:
        center = st.session_state["pin"] or ((sum(p[0] for p in gps_pts) / len(gps_pts), sum(p[1] for p in gps_pts) / len(gps_pts)) if gps_pts else (20.0, 0.0))
        zoom = 16 if (st.session_state["pin"] or gps_pts) else 2
        m = folium.Map(location=list(center), zoom_start=zoom, tiles=None, control_scale=True)
        folium.TileLayer("OpenStreetMap", name="Map", show=False).add_to(m)
        folium.TileLayer(ESRI, name="Satellite (Esri)", attr="Imagery © Esri, Maxar, Earthstar Geographics").add_to(m)
        for lat, lon, n in gps_pts:
            folium.CircleMarker([lat, lon], radius=5, color="#2563eb", fill=True, fill_opacity=.9, tooltip=n).add_to(m)
        if st.session_state["polygon"]:
            folium.Polygon(st.session_state["polygon"], color="#16a34a", weight=3, fill=True, fill_opacity=.12).add_to(m)
        if st.session_state["pin"]:
            folium.Marker(list(st.session_state["pin"]), icon=folium.Icon(color="red", icon="leaf"), tooltip="Field pin").add_to(m)
        Draw(export=False, draw_options={"polygon": True, "rectangle": True, "marker": False, "circle": False,
                                         "polyline": False, "circlemarker": False}, edit_options={"edit": False}).add_to(m)
        folium.LayerControl().add_to(m)
        out = st_folium(m, height=430, use_container_width=True, key="fr_map", returned_objects=["last_clicked", "all_drawings"])

    click = (out or {}).get("last_clicked")
    if click and (click["lat"], click["lng"]) != st.session_state["last_click"]:
        st.session_state["last_click"] = (click["lat"], click["lng"])
        st.session_state["pin"] = (round(click["lat"], 6), round(click["lng"], 6))
        st.session_state["pin_lat_in"], st.session_state["pin_lon_in"] = st.session_state["pin"]
    for feat in (out or {}).get("all_drawings") or []:
        g = feat.get("geometry", {})
        if g.get("type") == "Polygon" and g["coordinates"]:
            poly = [(lat, lon) for lon, lat in g["coordinates"][0]][:-1]
            if len(poly) >= 3 and poly != st.session_state["polygon"]:
                st.session_state["polygon"] = poly

    with rm:
        st.markdown("**Field pin**")
        st.session_state.setdefault("pin_lat_in", 0.0)
        st.session_state.setdefault("pin_lon_in", 0.0)
        lat_in = st.number_input("Latitude", min_value=-90.0, max_value=90.0, format="%.6f", key="pin_lat_in")
        lon_in = st.number_input("Longitude", min_value=-180.0, max_value=180.0, format="%.6f", key="pin_lon_in")
        if (lat_in or lon_in) and (lat_in, lon_in) != (st.session_state["pin"] or (0.0, 0.0)):
            st.session_state["pin"] = (lat_in, lon_in)
            st.rerun()
        if st.session_state["pin"]:
            st.caption(f"Pin: {st.session_state['pin'][0]:.5f}, {st.session_state['pin'][1]:.5f}")
        elif gps_pts:
            st.info(f"No pin placed. The median of the {len(gps_pts)} photo GPS points will be used.")
        else:
            st.warning("Place a pin, or upload original photos that carry GPS.")
        poly = st.session_state["polygon"]
        if poly:
            st.success(f"Boundary drawn: {geo.polygon_area_ha(poly):.1f} ha")
        manual_area = st.number_input("Field area, ha (if no boundary)", min_value=0.0, max_value=100000.0, value=0.0, step=1.0,
                                      key="f_area_ha", disabled=bool(poly))
        st.button("Clear pin and boundary", width="stretch", on_click=clear_location)

# --------------------------------------------------------------------------- 4 soil
with st.container(border=True):
    step(4, "Soil", "optional. A soil test gives much better advice than the map estimate")
    s1, s2, s3, s4 = st.columns(4)
    ph = s1.number_input("pH", min_value=0.0, max_value=10.5, value=0.0, step=0.1, key="s_ph", help="0 means not known")
    om = s2.number_input("Organic matter, %", min_value=0.0, max_value=30.0, value=0.0, step=0.1, key="s_om", help="0 means not known")
    tex = s3.selectbox("Texture", TEXTURES, key="s_tex")
    tdate = s4.date_input("Test date", value=None, max_value=date.today(), key="s_date")
    s5, s6, s7 = st.columns(3)
    pl = s5.selectbox("Phosphorus (lab rating)", LEVELS, key="s_p", help="Use the rating on your lab report so units and methods never get mixed up.")
    kl = s6.selectbox("Potassium (lab rating)", LEVELS, key="s_k")
    nl = s7.selectbox("Nitrogen (lab rating)", LEVELS, key="s_n")

with st.expander("Advanced settings"):
    a1, a2, a3, a4 = st.columns(4)
    conf = a1.slider("Counting confidence", 0.10, 0.90, 0.25, 0.05, key="a_conf",
                     help="0.25 gave the most accurate counts on our labelled photos.")
    kernels = a2.number_input("Kernels per head (0 = default)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="a_k")
    tkw = a3.number_input("Grams per 1000 kernels (0 = default)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="a_t")
    ref_lo = a4.number_input("Reference heads/m² low (0 = default)", min_value=0.0, max_value=2000.0, value=0.0, step=10.0, key="a_rl")
    ref_hi = a4.number_input("Reference heads/m² high (0 = default)", min_value=0.0, max_value=2000.0, value=0.0, step=10.0, key="a_rh")
    st.caption("Defaults for kernels, grain weight and the reference band are generic placeholders. Replace them with local values.")

# --------------------------------------------------------------------------- run
run_col, clr_col = st.columns([1, 4])
go = run_col.button("Analyze field", type="primary", width="stretch", disabled=not raws)
go = bool(st.session_state.pop("fr_rerun", False) and raws) or go      # set when chat answers are added to the form
if not raws:
    clr_col.caption("Upload at least one photo to begin.")


def build_field() -> FieldInput:
    return FieldInput(
        name=name or "My field", variety=variety, sowing_date=sowing, growth_stage=stage, irrigated=irrigated,
        previous_crop=previous, inputs_applied=applied, problems_noticed=problems,
        target_yield_t_ha=target or None,
        lat=st.session_state["pin"][0] if st.session_state["pin"] else None,
        lon=st.session_state["pin"][1] if st.session_state["pin"] else None,
        polygon=st.session_state["polygon"], area_ha=(manual_area or None) if not st.session_state["polygon"] else None,
        photo_area_m2=photo_area, kernels_per_head=kernels or None, tkw_g=tkw or None,
        reference_heads_low=ref_lo or None, reference_heads_high=ref_hi or None, conf=conf,
        soil=SoilInput(ph=ph or None, organic_matter_pct=om or None, texture=tex, p_level=pl, k_level=kl, n_level=nl,
                       test_date=tdate or None))


def input_signature() -> str:
    """A fingerprint of everything the report depends on, to notice edits made after the report was produced."""
    try:
        body = build_field().model_dump_json()
    except ValidationError:
        body = "invalid"
    return hashlib.md5((body + "|".join(digests) + repr(hand)).encode(), usedforsecurity=False).hexdigest()


if go:
    try:
        field = build_field()
    except ValidationError as e:
        st.error("Some details are not valid: " + "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()))
        st.stop()
    photos = [PhotoIn(name=n, raw=r, hand_count=h) for (n, r), h in zip(raws, hand)]
    app = graph.build_graph()
    state: dict = {}
    with st.status("Analyzing the field…", expanded=True) as status:
        try:
            for upd in app.stream({"field": field, "photos": photos, "status": "running"},
                                  {"configurable": {"detector": detector}}, stream_mode="updates"):
                for node, out_ in upd.items():
                    st.write(f"✓ {graph.NODE_LABELS.get(node, node)}")
                    state.update(out_ or {})
        except Exception as e:  # noqa: BLE001
            status.update(label="Something went wrong", state="error")
            st.exception(e)
            st.stop()
        if state.get("status") == "done":
            status.update(label="Report ready", state="complete", expanded=False)
        else:
            status.update(label="More information is needed", state="error", expanded=False)
    st.session_state["fr_sig"] = input_signature()
    st.session_state["fr_result"] = state.get("report")
    st.session_state["fr_blocked"] = None if state.get("status") == "done" else state.get("issues", [])
    st.session_state["fr_graph_mermaid"] = app.get_graph().draw_mermaid()

if st.session_state.get("fr_blocked"):
    st.markdown("### I need a bit more")
    for i in st.session_state["fr_blocked"]:
        st.markdown(f'<div class="issue {i["level"]}">{esc(i["message"])}</div>', unsafe_allow_html=True)

R = st.session_state.get("fr_result")
if not R:
    st.stop()
if st.session_state.get("fr_sig") != input_signature():
    st.warning("You changed the form or the photos after this report was made. Click **Analyze field** to update it.")


# =========================================================================== results
def to_markdown(r: dict) -> str:
    ev = r["evidence"]
    lines = [f"# Field report: {r['field']['name']}", "",
             f"**Verdict:** {r['verdict']} · **Confidence:** {r['confidence']} · generated {r['generated']}", "", r["summary"], "",
             "## Why", ""]
    lines += [f"- {x['text']} ({', '.join(x['evidence_ids'])})" for x in r["reasons"]]
    soon, soon_actions = r["packet"].get("outlook", []), [a for a in r["actions"] if a["window"] == "next_days"]
    if soon or soon_actions:
        lines += ["", "## Coming up in the next few days (forecast, less certain than measurements)", ""]
        lines += [f"- {o['text']}" for o in soon]
        lines += [f"- {a['text']}" + (" *(confirm with an agronomist)*" if a["confirm"] else "") + f"\n  - {a['why']}" for a in soon_actions]
    for title, win in (("This season", "this_season"), ("Next season", "next_season")):
        items = [a for a in r["actions"] if a["window"] == win]
        if items:
            lines += ["", f"## {title}", ""]
            lines += [f"- {a['text']}" + (" *(confirm with an agronomist)*" if a["confirm"] else "") + f"\n  - {a['why']}" for a in items]
    if r["open_questions"]:
        lines += ["", "## Still open", ""] + [f"- {q}" for q in r["open_questions"]]
    lines += ["", "## Assumptions", ""] + [f"- {a}" for a in r["assumptions"]]
    if r["gaps"]:
        lines += ["", "## Missing information", ""] + [f"- {g['message']}" for g in r["gaps"]]
    lines += ["", "## Evidence", "", "| id | item | value | note |", "|---|---|---|---|"]
    lines += [f"| {e['id']} | {e['label']} | {e['value']} {e['unit']} | {e['note']} |" for e in ev.values()]
    lines += ["", f"_{r['limitations']}_"]
    return "\n".join(lines)


cls = {"Good": "good", "Watch": "watch", "Poor": "poor"}.get(R["verdict"], "none")
st.markdown(
    f'<div class="verdict {cls}"><div class="vl">{esc(R["verdict"])}</div><div class="vs">'
    f'{esc(R["confidence"])} confidence · {esc(R["field"]["name"])} · {esc(R["field"]["growth_stage"])}'
    f'{" · " + esc(R["location"].get("label", "")) if R["location"].get("label") else ""}</div></div>', unsafe_allow_html=True)
if R["needs_review"]:
    st.warning("This report should be reviewed by an agronomist before anyone acts on it "
               "(the verdict is Poor, or confidence is low).")

agg, y = R["aggregate"], R["yield"]
k1, k2, k3, k4, k5 = st.columns(5, gap="small")
kp = [
    (k1, "Heads per m²", f"{agg['density_mean']:.0f}" if "density_mean" in agg else "—",
     f"95% interval {agg['ci_low']:.0f} to {agg['ci_high']:.0f}" if "density_mean" in agg else "needs the photo area", True),
    (k2, "Yield forecast", f"{y['central']:.1f} t/ha" if y else "—", f"range {y['low']:.1f} to {y['high']:.1f}" if y else "needs the photo area", False),
    (k3, "Whole field", next((f"{e['value']} t" for e in R["evidence"].values() if e["label"].startswith("Whole-field")), "—"),
     next((e["note"] for e in R["evidence"].values() if e["label"].startswith("Whole-field")), "needs boundary or area"), False),
    (k4, "Variation", f"{agg['cv_pct']:.0f}%" if "cv_pct" in agg else "—", "between photo points", False),
    (k5, "Photos used", f"{agg['n_used']}", f"of {agg['n_photos']} uploaded", False),
]
for col, k, v, s, acc in kp:
    col.markdown(f'<div class="stat {"accent" if acc else ""}"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{esc(s)}</div></div>',
                 unsafe_allow_html=True)
st.write("")

view = st.segmented_control("View", ["Report", "Map and photos", "Weather and soil", "Evidence", "Run details"],
                            default="Report", required=True, key="fr_view", label_visibility="collapsed")


def action_card(a: dict) -> None:
    tag = '<span class="tag">CONFIRM WITH AN AGRONOMIST</span>' if a["confirm"] else ""
    st.markdown(f'<div class="act"><b>{esc(a["text"])}{tag}</b><div class="why">{esc(a["why"])} '
                f'{ev_chips(a["evidence_ids"], R["evidence"])}</div></div>', unsafe_allow_html=True)


if view == "Report":
    st.markdown(f"#### Summary\n{R['summary']}")
    st.markdown("#### Why")
    for x in R["reasons"]:
        st.markdown(f"- {esc(x['text'])} {ev_chips(x['evidence_ids'], R['evidence'])}", unsafe_allow_html=True)
    if "forecast" in R:                       # reports made before the forecast existed have no such section
        fc = R["forecast"] or {}
        st.markdown("#### Coming up in the next few days")
        if fc.get("summary"):
            age = wxmod.forecast_age_hours(fc)
            if age is not None and age > acfg.FORECAST_STALE_HOURS:
                st.warning(f"This forecast was made about {age:.0f} hours ago. Click **Analyze field** to get a fresh one.")
            for o in R["packet"].get("outlook", []):
                st.markdown(f"- {esc(o['text'])} {ev_chips(o['evidence'], R['evidence'])}", unsafe_allow_html=True)
            for a in [a for a in R["actions"] if a["window"] == "next_days"]:
                action_card(a)
            st.caption(f"This is a forecast, so it can be wrong. Only the next {fc['summary']['action_days']} days drive this advice, "
                       "and it does not change the yield range.")
        else:
            st.caption("The weather forecast could not be loaded, so there is no advice for the next few days.")
    ca, cb = st.columns(2, gap="large")
    for col, title, win in ((ca, "This season", "this_season"), (cb, "Next season", "next_season")):
        with col:
            st.markdown(f"#### {title}")
            items = [a for a in R["actions"] if a["window"] == win]
            if not items:
                st.caption("Nothing specific.")
            for a in items:
                action_card(a)
    if R["open_questions"]:
        st.markdown("#### Still open")
        st.caption("Answer these in the chat at the bottom of the page, or fill in the form above and analyze again.")
        for qi, q in enumerate(R["open_questions"]):
            qc1, qc2 = st.columns([5, 1.3])
            qc1.markdown(f"- {q}")
            qc2.button("Answer in chat", key=f"oq_{qi}", on_click=ask_in_chat, args=(q,), width="stretch")
    if R["gaps"] or R["issues"]:
        st.markdown("#### What would make this better")
        for g in R["gaps"]:
            st.markdown(f'<div class="gap">• {esc(g["message"])}</div>', unsafe_allow_html=True)
        for i in R["issues"]:
            st.markdown(f'<div class="gap">• {esc(i["message"])}</div>', unsafe_allow_html=True)
    with st.expander("Assumptions and limits"):
        for a in R["assumptions"]:
            st.markdown(f"- {a}")
        st.caption(R["limitations"])
    d1, d2, _ = st.columns([1, 1, 3])
    d1.download_button("⬇  Report (Markdown)", to_markdown(R), file_name="field_report.md", mime="text/markdown", width="stretch")
    d2.download_button("⬇  Evidence (JSON)", json.dumps(R["packet"], indent=1, ensure_ascii=False), file_name="evidence.json",
                       mime="application/json", width="stretch")

elif view == "Map and photos":
    pts = [p for p in R["photos"] if p.get("usable") and p.get("lat") is not None]
    loc = R["location"]
    mm = folium.Map(location=[loc["lat"], loc["lon"]], zoom_start=17, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="Map", show=False).add_to(mm)
    folium.TileLayer(ESRI, name="Satellite (Esri)", attr="Imagery © Esri, Maxar, Earthstar Geographics", show=True).add_to(mm)
    meand = agg.get("density_mean")
    for p in pts:
        d = p.get("density_m2")
        col = "#2563eb" if d is None or not meand else ("#dc2626" if d < 0.85 * meand else "#16a34a" if d > 1.1 * meand else "#d97706")
        txt = f"{p['name']}: {p['count']:.0f} heads" + (f", {d:.0f} per m²" if d is not None else "")
        folium.CircleMarker([p["lat"], p["lon"]], radius=9, color="#fff", weight=2, fill=True, fill_color=col, fill_opacity=.95, tooltip=txt).add_to(mm)
    if R["field"].get("polygon"):
        folium.Polygon(R["field"]["polygon"], color="#16a34a", weight=3, fill=True, fill_opacity=.1).add_to(mm)
    folium.Marker([loc["lat"], loc["lon"]], icon=folium.Icon(color="red", icon="leaf"), tooltip="Field").add_to(mm)
    bounds = [[p["lat"], p["lon"]] for p in pts] + ([list(c) for c in R["field"]["polygon"]] if R["field"].get("polygon") else [])
    if len(bounds) >= 2:
        mm.fit_bounds(bounds, padding=(30, 30))
    folium.LayerControl().add_to(mm)
    st_folium(mm, height=440, use_container_width=True, key="fr_result_map", returned_objects=[])
    st.caption("Photo points are coloured by head density: green is above the field average, amber near it, red well below.  "
               + loc.get("accuracy_note", ""))
    lc = st.columns(4)
    lc[0].metric("Location confidence", loc.get("confidence", "—").capitalize())
    lc[1].metric("Photos with GPS", f"{loc.get('n_gps', 0)} of {agg['n_photos']}")
    if "area_ha" in loc:
        lc[2].metric("Boundary area", f"{loc['area_ha']:.1f} ha")
    if "spread_m" in loc:
        lc[3].metric("Photo spread", f"{loc['spread_m']} m")
    if R["spatial"] and R["spatial"].get("gradient"):
        g = R["spatial"]["gradient"]
        st.info(f"Head density falls toward the {g['away_from']}, by about {g['change_pct']}% across the sampled area.")

    st.markdown("#### Photos")
    ok = [p for p in R["photos"] if p.get("usable")]
    cols = st.columns(3)
    for i, p in enumerate(ok):
        with cols[i % 3]:
            st.image(p["annotated"], width="stretch")
            dens = f" · {p['density_m2']:.0f} per m²" if p.get("density_m2") is not None else ""
            flags = f" · ⚠ {', '.join(p['quality']['flags'])}" if p["quality"]["flags"] else ""
            st.caption(f"**{p['name']}** · {p['count']:.0f} heads{dens}{flags}")
    bad = [p for p in R["photos"] if not p.get("usable")]
    for p in bad:
        st.markdown(f'<div class="issue warning">{esc(p["name"])}: {esc(p.get("error", "could not be read"))}</div>', unsafe_allow_html=True)

elif view == "Weather and soil":
    wx = R["weather"]
    if wx and "summary" in wx:
        s = wx["summary"]
        w1, w2, w3, w4, w5 = st.columns(5)
        w1.metric("Growing degree days", f"{s['gdd_since_sowing']:,}")
        w2.metric("Rain since sowing", f"{s['rain_since_sowing_mm']} mm")
        w3.metric("Rain minus crop water use (4 wk)", f"{s['water_balance_28d_mm']} mm")
        w4.metric("Hot days ≥30 °C (3 wk)", s["heat_days_21d"])
        w5.metric("Rainy days (2 wk)", s["rainy_days_14d"])
        wk = pd.DataFrame(wx["weekly"]).set_index("week")
        st.markdown("#### Weekly rain and crop water use (mm)")
        st.bar_chart(wk[["rain_mm", "et0_mm"]].rename(columns={"rain_mm": "Rain", "et0_mm": "Reference water use"}), height=260)
        st.markdown("#### Weekly temperature (°C)")
        st.line_chart(wk[["tmax", "tmin"]].rename(columns={"tmax": "Highest", "tmin": "Lowest"}), height=220)
        st.caption(f"Source: {wx['source']}, data to {s['last_date']}. Water use is FAO reference evapotranspiration.")
    else:
        st.info("No weather data for this run." + (f" ({wx['error']})" if wx and "error" in wx else ""))
    fc = R.get("forecast") or {}
    st.markdown("#### Forecast for the next days")
    if fc.get("days"):
        n_act = acfg.FORECAST_ACTION_DAYS
        st.dataframe(pd.DataFrame([{
            "Day": date.fromisoformat(d["date"]).strftime("%a %d %b") + ("" if i < n_act else "  (less certain)"),
            "Rain, mm": d["rain_mm"], "Chance of rain, %": d["chance_pct"], "High, °C": d["tmax"], "Low, °C": d["tmin"],
            "Humidity, %": d["humidity_pct"]} for i, d in enumerate(fc["days"])]), hide_index=True, width="stretch")
        st.bar_chart(pd.DataFrame({"Forecast rain (mm)": [d["rain_mm"] or 0 for d in fc["days"]]},
                                  index=[d["date"] for d in fc["days"]]), height=200)
        age = wxmod.forecast_age_hours(fc)
        st.caption(f"Source: {fc['source']}" + ("" if age is None else ", made just now" if age < 1 else f", made {age:.0f} hours ago")
                   + f". The first {n_act} days drive the advice. Later days are for planning only.")
    else:
        st.info("No forecast for this run." + (f" ({fc['error']})" if fc.get("error") else ""))
    st.markdown("#### Soil")
    so = R["soil"]
    st.dataframe(pd.DataFrame({"Item": ["Source", "pH", "Organic matter %", "Texture", "Phosphorus", "Potassium", "Nitrogen", "Test date"],
                               "Value": [so["source"], so["ph"], so["organic_matter_pct"], so["texture"], so["p_level"], so["k_level"], so["n_level"], so["test_date"]]}
                              ).astype(str), hide_index=True, width="stretch")
    if so.get("estimate_error"):
        st.caption("Soil map service: " + so["estimate_error"])

elif view == "Evidence":
    st.caption("Every number in the report comes from this list. The language model may cite these ids but cannot add figures.")
    st.dataframe(pd.DataFrame(list(R["evidence"].values())).rename(columns={"id": "Id", "label": "Item", "value": "Value", "unit": "Unit", "note": "Note"}
                                                                   ).astype({"Value": str}), hide_index=True, width="stretch")
    st.markdown("#### Drivers behind the verdict")
    sev = {0: "none", 1: "minor", 2: "major"}
    st.dataframe(pd.DataFrame([{"Id": d["id"], "Driver": d["label"], "Severity": sev[d["severity"]], "Detail": d["text"],
                                "Evidence": ", ".join(d["evidence"])} for d in R["drivers"]]), hide_index=True, width="stretch")

else:
    stt = R["stats"]
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Model", stt["model"])
    r2.metric("Attempts", stt["attempts"])
    r3.metric("Report written by", "Fallback" if stt["used_fallback"] else "AI model",
              help="AI model: the local language model wrote the wording and it passed every check. "
                   "Fallback: the rule-based wording was used because the model's reply failed the checks twice, or it was not available.")
    r4.metric("Model time", f"{stt['seconds']:.0f} s")
    st.caption(f"Tokens: {stt['prompt_tokens']:,} in, {stt['output_tokens']:,} out.")
    if stt["errors"]:
        st.markdown("**Checks the draft failed before it was accepted or replaced**")
        for e in dict.fromkeys(stt["errors"]):
            st.markdown(f"- {e}")
    st.markdown("**Time per workflow step (seconds)**")
    st.dataframe(pd.DataFrame({"Step": list(R["timings"]), "Seconds": list(R["timings"].values())}), hide_index=True, width="stretch")
    if st.session_state.get("fr_graph_mermaid"):
        with st.expander("Workflow graph (Mermaid)"):
            st.code(st.session_state["fr_graph_mermaid"], language="text")


# =========================================================================== chat
st.divider()
st.markdown("### Ask about this report")
st.caption("Ask why the verdict is what it is, or answer the open questions in your own words. "
           "Facts you give can be added to the report with one click.")
history = st.session_state.setdefault("fr_chat", [])
for ci, m in enumerate(history):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("updates"):
            st.markdown("**I can add this to the report:**")
            for line in chat.describe(m["updates"]):
                st.markdown(f"- {line}")
            if m.get("applied"):
                st.caption("✓ Added to the form. The report was run again.")
            else:
                st.button("Add to the report and run it again", key=f"apply_{ci}", type="primary",
                          on_click=apply_updates, args=(m["updates"], ci))
if history:
    st.button("Clear chat", key="clear_chat", on_click=lambda: st.session_state.update(fr_chat=[]))

user_msg = st.chat_input("Ask a question, or answer one of the open questions…")
if user_msg:
    history.append({"role": "user", "content": user_msg})
    with st.spinner("Thinking…"):
        res = chat.chat_turn(R["packet"], [{"role": h["role"], "content": h["content"]} for h in history[:-1]], user_msg)
    history.append({"role": "assistant", "content": res["answer"], "updates": res["updates"]})
    st.rerun()
