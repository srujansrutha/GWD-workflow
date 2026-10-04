"""Wheat Head Counter - upload field photos, see the detected wheat heads and their count.

Run (from this folder):  streamlit run app.py        or double-click run_app.bat
The model is loaded onto the GPU once and stays there until the Streamlit process is stopped.
"""
from __future__ import annotations

import hashlib
import html
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import wheat_detector as wd

st.set_page_config(page_title="Wheat Head Counter", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- styling
st.markdown(
    """
<style>
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1400px}
header[data-testid="stHeader"]{background:transparent}
footer{visibility:hidden}

.hero{background:linear-gradient(120deg,#052e16 0%,#14532d 38%,#3f6212 100%);border-radius:20px;padding:30px 36px;
      color:#fff;margin-bottom:1.2rem;box-shadow:0 8px 30px rgba(5,46,22,.28)}
.hero h1{font-size:2.3rem;line-height:1.15;margin:0 0 .35rem 0;padding:0;color:#fff;font-weight:800;letter-spacing:-.02em}
.hero p{margin:0 0 1rem 0;font-size:1.04rem;opacity:.88;max-width:760px}
.chip{display:inline-flex;align-items:center;gap:7px;padding:5px 13px;margin:0 8px 6px 0;border-radius:999px;
      background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.26);font-size:.8rem;font-weight:500;color:#fff}
.dot{width:8px;height:8px;border-radius:50%;background:#4ade80;box-shadow:0 0 0 3px rgba(74,222,128,.28)}
.dot.warn{background:#fbbf24;box-shadow:0 0 0 3px rgba(251,191,36,.28)}

[data-testid="stFileUploaderDropzone"]{border:2px dashed rgba(22,163,74,.6);border-radius:18px;padding:2.4rem 1rem;
      background:rgba(22,163,74,.06);transition:background .2s,border-color .2s}
[data-testid="stFileUploaderDropzone"]:hover{background:rgba(22,163,74,.12);border-color:#16a34a}

.stat{border:1px solid rgba(128,128,128,.28);border-radius:16px;padding:16px 20px;background:rgba(128,128,128,.07);height:100%}
.stat .k{font-size:.7rem;letter-spacing:.09em;text-transform:uppercase;opacity:.65;font-weight:700}
.stat .v{font-size:2.15rem;font-weight:800;line-height:1.18}
.stat .s{font-size:.78rem;opacity:.6}
.stat.accent{background:linear-gradient(135deg,rgba(22,163,74,.24),rgba(132,204,22,.12));border-color:rgba(22,163,74,.55)}
.stat.accent .v{color:#16a34a}

.img-title{font-weight:700;font-size:1.05rem;word-break:break-all}
.img-sub{opacity:.6;font-size:.82rem}
.pill{display:inline-block;padding:6px 18px;border-radius:999px;background:#16a34a;color:#fff;font-weight:800;font-size:1.1rem;
      box-shadow:0 2px 10px rgba(22,163,74,.35)}

.step{border:1px solid rgba(128,128,128,.28);border-radius:16px;padding:18px 20px;background:rgba(128,128,128,.06);height:100%}
.step .n{display:inline-flex;width:30px;height:30px;border-radius:50%;background:#16a34a;color:#fff;font-weight:800;
         align-items:center;justify-content:center;margin-bottom:8px}
.step b{display:block;margin-bottom:2px}
.step span{opacity:.72;font-size:.9rem}
.errcard{border:1px solid rgba(239,68,68,.55);background:rgba(239,68,68,.08);border-radius:14px;padding:12px 16px;margin-bottom:.6rem}
</style>
""",
    unsafe_allow_html=True,
)

# Measured on 2,574 labelled photos at confidence 0.25 (predicted count vs. hand-labelled count).
ACCURACY = pd.DataFrame(
    {
        "Photos": ["Farms the model trained on", "Held-out farms (usask, ethz)", "Farms from other countries"],
        "Images": [247, 947, 1380],
        "Typical count error": ["6.0%", "4.6%", "13.2%"],
        "Average bias": ["+3.0%", "+1.7%", "−14.0%"],
        "Correlation (r)": [0.98, 0.97, 0.95],
    }
)


# --------------------------------------------------------------------------- model + cached work
@st.cache_resource(show_spinner="Loading the model onto the GPU…")
def load_detector() -> wd.Detector:
    """Runs once per server process; the model then stays resident on the GPU until the app is stopped."""
    weights = wd.find_weights()
    if weights is None:
        raise FileNotFoundError(
            "No model weights found. Put the file at weights/wheat_yolo11s_gwhd21.pt "
            "or set the WHEAT_MODEL environment variable."
        )
    return wd.Detector(weights)


@st.cache_data(show_spinner=False, max_entries=200)
def run_detection(digest: str, _raw: bytes, imgsz: int, iou: float):
    """Decode + detect once per (image, resolution, overlap). The confidence slider then filters the result."""
    dec = wd.decode_image(_raw)
    det = load_detector().detect(dec.image, imgsz=imgsz, iou=iou)
    return dec.fmt, dec.mode, det


@st.cache_data(show_spinner=False, max_entries=300)
def render_annotated(digest: str, _raw: bytes, imgsz: int, iou: float, conf: float, style: str, color: str,
                     thickness: float, show_conf: bool) -> bytes:
    dec = wd.decode_image(_raw)
    _, _, det = run_detection(digest, _raw, imgsz, iou)
    boxes, scores = det.at(conf)
    return wd.to_jpeg(wd.annotate(dec.image, boxes, scores, color=color, style=style, thickness=thickness,
                                  show_conf=show_conf))


@st.cache_data(show_spinner=False, max_entries=100)
def render_original(digest: str, _raw: bytes) -> bytes:
    img = wd.decode_image(_raw).image
    scale = wd.display_scale(img.width, img.height)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    return wd.to_jpeg(img, 90)


@dataclass
class Record:
    name: str
    digest: str
    raw: bytes
    fmt: str
    mode: str
    det: wd.Detection
    count: int
    mean_conf: float


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def plural(n: int) -> str:
    return "1 wheat head" if n == 1 else f"{n:,} wheat heads"


# --------------------------------------------------------------------------- load model + header
try:
    detector = load_detector()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load the model: {e}")
    st.stop()

dot = "dot" if detector.on_gpu else "dot warn"
st.markdown(
    f"""
<div class="hero">
  <h1>🌾 Wheat Head Counter</h1>
  <p>Upload field photos and get every wheat head detected, outlined and counted — in seconds, on your own GPU.</p>
  <span class="chip"><span class="{dot}"></span>{esc(detector.device_name)}</span>
  <span class="chip">Model on {esc(detector.model_device)}</span>
  <span class="chip">YOLO11s · {esc(detector.weights.name)}</span>
  <span class="chip">Loaded in {detector.load_seconds:.1f}s</span>
</div>
""",
    unsafe_allow_html=True,
)
if not detector.on_gpu:
    st.warning("No GPU was found, so detection runs on the CPU and will be much slower.")

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### Detection")
    conf = st.slider(
        "Confidence threshold", 0.10, 0.90, 0.25, 0.05,
        help="Lower = finds more heads but also more false ones. 0.25 gave the most accurate counts on our "
             "labelled test photos. Changing this does not re-run the model.",
    )
    with st.expander("Advanced"):
        imgsz = st.select_slider(
            "Detection resolution", options=[1024, 1280, 1536], value=1024,
            help="1024 is what the model was trained at. Higher can help with very small heads but is slower "
                 "and may reduce accuracy.",
        )
        iou = st.slider(
            "Merge overlapping boxes", 0.30, 0.90, 0.70, 0.05,
            help="Boxes overlapping more than this are treated as the same head. Lower merges more aggressively.",
        )
    st.markdown("### Appearance")
    style = st.segmented_control("Box style", ["Outline", "Filled"], default="Outline", required=True)
    color = st.color_picker("Box colour", "#22c55e")
    thickness = st.slider("Line thickness", 0.5, 4.0, 1.5, 0.5)
    show_conf = st.toggle("Show confidence on boxes", value=False)

    st.divider()
    mem = detector.gpu_memory_gb()
    st.caption(
        f"**Model:** `{detector.weights.name}`  \n"
        f"**Device:** {detector.device_name}  \n"
        + (f"**GPU memory in use:** {mem:.2f} GB" if mem is not None else "")
    )

# --------------------------------------------------------------------------- upload
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

uploads = st.file_uploader(
    "Upload wheat field photos",
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}",
    label_visibility="collapsed",
    help="JPG, PNG, TIFF, WEBP, BMP, GIF, HEIC and more. You can select many files at once.",
)
st.caption("Drop one or many photos above · JPG · PNG · TIFF · WEBP · BMP · GIF · HEIC and other common formats · up to 200 MB each")

# --------------------------------------------------------------------------- empty state
if not uploads:
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="step"><div class="n">1</div><b>Upload</b><span>Drag in one photo or a whole folder of field images.</span></div>', unsafe_allow_html=True)
    c2.markdown('<div class="step"><div class="n">2</div><b>Detect</b><span>The model finds every wheat head and draws a box around it.</span></div>', unsafe_allow_html=True)
    c3.markdown('<div class="step"><div class="n">3</div><b>Review &amp; export</b><span>See counts per image, then download annotated pictures and a CSV.</span></div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown("##### Tips for the best counts")
        st.markdown(
            "- Use **overhead or near-overhead** photos of the crop, like the training images.\n"
            "- Heads should be roughly **40–130 px wide** when the photo is shown at 1024 px across.\n"
            "- Sharp, evenly lit photos work better than blurry or backlit ones.\n"
            "- Check one or two images by eye before trusting counts from a new camera or crop."
        )
    with right:
        st.markdown("##### How accurate is it?")
        st.dataframe(ACCURACY.drop(columns=["Correlation (r)"]), hide_index=True, width="stretch")
        st.caption("Predicted vs. hand-labelled counts at confidence 0.25. Photos from unfamiliar farms, cameras or growth "
                   "stages tend to be undercounted by about 15%.")
    st.stop()

# --------------------------------------------------------------------------- detect
if len(uploads) > 40:
    st.warning(f"{len(uploads)} images selected — this may take a while and the page can get heavy. Consider batches of 40 or fewer.")

records: list[Record] = []
errors: list[tuple[str, str]] = []
with st.spinner(f"Detecting wheat heads in {len(uploads)} image{'s' if len(uploads) != 1 else ''}…"):
    for up in uploads:
        raw = up.getvalue()
        digest = hashlib.md5(raw).hexdigest()
        try:
            fmt, mode, det = run_detection(digest, raw, imgsz, iou)
        except (wd.ImageReadError, MemoryError) as e:
            errors.append((up.name, str(e)))
            continue
        boxes, scores = det.at(conf)
        records.append(Record(up.name, digest, raw, fmt, mode, det, len(boxes),
                              float(scores.mean()) if len(scores) else 0.0))

for name, msg in errors:
    st.markdown(f'<div class="errcard"><b>{esc(name)}</b> — {esc(msg)}</div>', unsafe_allow_html=True)

if not records:
    st.info("None of the uploaded files could be read as an image.")
    st.stop()

# --------------------------------------------------------------------------- summary
total = sum(r.count for r in records)
avg = total / len(records)
all_conf = [r.mean_conf for r in records if r.count]
mean_conf = float(np.mean(all_conf)) if all_conf else 0.0

top = st.columns([1, 1.25, 1, 1], gap="medium")
cards = [
    ("Images", f"{len(records)}", "processed" + (f" · {len(errors)} skipped" if errors else ""), ""),
    ("Total wheat heads", f"{total:,}", f"at confidence ≥ {conf:.2f}", "accent"),
    ("Average per image", f"{avg:,.1f}", f"range {min(r.count for r in records):,} – {max(r.count for r in records):,}", ""),
    ("Average confidence", f"{mean_conf:.0%}", "of detected heads", ""),
]
for col, (k, v, s, cls) in zip(top, cards):
    col.markdown(f'<div class="stat {cls}"><div class="k">{k}</div><div class="v">{v}</div><div class="s">{s}</div></div>',
                 unsafe_allow_html=True)

st.write("")
table = pd.DataFrame(
    {
        "Image": [r.name for r in records],
        "Wheat heads": [r.count for r in records],
        "Avg confidence": [round(r.mean_conf, 3) for r in records],
        "Size": [f"{r.det.width}×{r.det.height}" for r in records],
        "Format": [r.fmt for r in records],
        "Time (ms)": [round(r.det.ms) for r in records],
    }
)
csv_df = table.rename(columns={"Image": "image", "Wheat heads": "wheat_heads", "Avg confidence": "mean_confidence",
                               "Size": "size", "Format": "format", "Time (ms)": "inference_ms"})
csv_df.insert(5, "confidence_threshold", conf)
csv_bytes = csv_df.to_csv(index=False).encode("utf-8")


def export_name(r: Record, i: int) -> str:
    return f"{Path(r.name).stem}_{r.count}heads.jpg"


def build_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for i, r in enumerate(records):
            data = render_annotated(r.digest, r.raw, imgsz, iou, conf, style, color, thickness, show_conf)
            z.writestr(f"{i + 1:02d}_{export_name(r, i)}", data)
        z.writestr("counts.csv", csv_bytes)
    return buf.getvalue()


b1, b2, b3, _ = st.columns([1.3, 1, 1, 2.2], gap="small")
b1.download_button("⬇  Annotated images (ZIP)", data=build_zip, file_name="wheat_detections.zip", mime="application/zip",
                   type="primary", width="stretch")
b2.download_button("⬇  Counts (CSV)", data=csv_bytes, file_name="wheat_counts.csv", mime="text/csv", width="stretch")
if b3.button("✕  Clear all", width="stretch"):
    st.session_state.uploader_key += 1
    st.rerun()

if len(records) > 1:
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "Wheat heads": st.column_config.ProgressColumn("Wheat heads", format="%d", min_value=0,
                                                           max_value=int(max(max(r.count for r in records), 1))),
            "Avg confidence": st.column_config.NumberColumn("Avg confidence", format="%.2f"),
        },
    )

# --------------------------------------------------------------------------- per-image results
st.markdown("### Results")
many = len(records) > 6
for i, r in enumerate(records):
    holder = (st.expander(f"{r.name} — {plural(r.count)}", expanded=i < 2) if many else st.container(border=True))
    with holder:
        if not many:
            h1, h2 = st.columns([4, 1.5])
            h1.markdown(f'<div class="img-title">{esc(r.name)}</div><div class="img-sub">{esc(r.fmt)} · {r.det.width}×{r.det.height} px</div>',
                        unsafe_allow_html=True)
            h2.markdown(f'<div style="text-align:right"><span class="pill">{plural(r.count)}</span></div>', unsafe_allow_html=True)

        left, right = st.columns([3, 1.15], gap="large")
        annotated = render_annotated(r.digest, r.raw, imgsz, iou, conf, style, color, thickness, show_conf)
        with left:
            tab_det, tab_orig = st.tabs(["Detected", "Original"])
            tab_det.image(annotated, width="stretch")
            tab_orig.image(render_original(r.digest, r.raw), width="stretch")
            scale = wd.display_scale(r.det.width, r.det.height)
            if scale < 1:
                st.caption(f"Large image: shown and downloaded at {round(r.det.width * scale)}×{round(r.det.height * scale)} px; "
                           "counting used the full image.")
        with right:
            st.metric("Wheat heads", f"{r.count:,}")
            st.metric("Average confidence", f"{r.mean_conf:.0%}" if r.count else "—")
            st.metric("Detection time", f"{r.det.ms:.0f} ms")
            _, sc = r.det.at(conf)
            if len(sc) >= 3:
                edges = np.linspace(conf, 1.0, 8)
                hist, _ = np.histogram(sc, bins=edges)
                labels = [f"{a:.2f}–{b:.2f}" for a, b in zip(edges[:-1], edges[1:])]
                st.caption("Confidence of detected heads")
                st.bar_chart(pd.DataFrame({"heads": hist}, index=labels), height=140, color="#16a34a")
            if r.det.floor > conf:
                st.caption(f"⚠ This image was cluttered; detection ran with a stricter floor of {r.det.floor:.2f}.")
            st.download_button("⬇  Download", data=annotated, file_name=export_name(r, i), mime="image/jpeg",
                               key=f"dl_{i}_{r.digest}", width="stretch")

with st.expander("How accurate are these counts?"):
    st.dataframe(ACCURACY, hide_index=True, width="stretch")
    st.caption(
        "Measured by comparing predicted counts with hand-labelled counts on 2,574 photos at confidence 0.25. "
        "“Typical count error” is the median absolute error per image; “bias” is whether the total is over (+) or under (−) "
        "counted. Your own photos may differ, especially with different cameras, crops or growth stages."
    )
