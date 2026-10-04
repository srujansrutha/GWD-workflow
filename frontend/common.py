"""Pieces shared by the pages: the GPU detector (loaded once per server process) and base styling."""
from __future__ import annotations

import streamlit as st

import wheat_detector as wd


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


REPORT_CSS = """
<style>
.block-container{padding-top:1.4rem;padding-bottom:3rem;max-width:1400px}
header[data-testid="stHeader"]{background:transparent}
footer{visibility:hidden}
.hero{background:linear-gradient(120deg,#052e16 0%,#14532d 38%,#3f6212 100%);border-radius:20px;padding:30px 36px;
      color:#fff;margin-bottom:1.2rem;box-shadow:0 8px 30px rgba(5,46,22,.28)}
.hero h1{font-size:2.3rem;line-height:1.15;margin:0 0 .35rem 0;padding:0;color:#fff;font-weight:800;letter-spacing:-.02em}
.hero p{margin:0 0 1rem 0;font-size:1.04rem;opacity:.88;max-width:780px}
.chip{display:inline-flex;align-items:center;gap:7px;padding:5px 13px;margin:0 8px 6px 0;border-radius:999px;
      background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.26);font-size:.8rem;font-weight:500;color:#fff}
.dot{width:8px;height:8px;border-radius:50%;background:#4ade80;box-shadow:0 0 0 3px rgba(74,222,128,.28)}
.dot.warn{background:#fbbf24;box-shadow:0 0 0 3px rgba(251,191,36,.28)}
.dot.bad{background:#f87171;box-shadow:0 0 0 3px rgba(248,113,113,.28)}

.step-h{display:flex;align-items:center;gap:12px;margin:.2rem 0 .6rem}
.step-n{display:inline-flex;width:30px;height:30px;border-radius:50%;background:#16a34a;color:#fff;font-weight:800;
        align-items:center;justify-content:center;flex:none}
.step-t{font-weight:800;font-size:1.15rem}
.step-s{opacity:.65;font-size:.85rem;margin-left:4px}

.stat{border:1px solid rgba(128,128,128,.28);border-radius:16px;padding:14px 18px;background:rgba(128,128,128,.07);height:100%}
.stat .k{font-size:.7rem;letter-spacing:.09em;text-transform:uppercase;opacity:.65;font-weight:700}
.stat .v{font-size:1.8rem;font-weight:800;line-height:1.2}
.stat .s{font-size:.78rem;opacity:.62}
.stat.accent{background:linear-gradient(135deg,rgba(22,163,74,.24),rgba(132,204,22,.12));border-color:rgba(22,163,74,.55)}

.verdict{display:flex;flex-wrap:wrap;align-items:center;gap:14px 22px;border-radius:18px;padding:18px 24px;margin:.4rem 0 1rem;border:1px solid}
.verdict .vl{font-size:2.1rem;font-weight:800;line-height:1.1;letter-spacing:-.01em}
.verdict .vs{opacity:.8;font-size:.95rem}
.verdict.good{background:rgba(22,163,74,.12);border-color:rgba(22,163,74,.55)} .verdict.good .vl{color:#16a34a}
.verdict.watch{background:rgba(217,119,6,.12);border-color:rgba(217,119,6,.55)} .verdict.watch .vl{color:#d97706}
.verdict.poor{background:rgba(220,38,38,.10);border-color:rgba(220,38,38,.55)} .verdict.poor .vl{color:#dc2626}
.verdict.none{background:rgba(128,128,128,.10);border-color:rgba(128,128,128,.45)}

.ev{display:inline-block;font-family:ui-monospace,Consolas,monospace;font-size:.72rem;font-weight:600;padding:1px 7px;
    margin:0 3px 0 0;border-radius:6px;background:rgba(22,163,74,.14);color:#16a34a;border:1px solid rgba(22,163,74,.35);cursor:help}
.act{border:1px solid rgba(128,128,128,.28);border-radius:14px;padding:12px 16px;margin-bottom:.6rem;background:rgba(128,128,128,.05)}
.act b{display:block;margin-bottom:2px}
.act .why{opacity:.78;font-size:.9rem}
.tag{display:inline-block;font-size:.68rem;font-weight:700;letter-spacing:.05em;padding:2px 9px;border-radius:999px;margin-left:6px;
     background:rgba(217,119,6,.15);color:#d97706;border:1px solid rgba(217,119,6,.4);vertical-align:middle}
.issue{border-radius:12px;padding:10px 14px;margin-bottom:.5rem;border:1px solid}
.issue.error{background:rgba(220,38,38,.08);border-color:rgba(220,38,38,.5)}
.issue.warning{background:rgba(217,119,6,.08);border-color:rgba(217,119,6,.45)}
.gap{opacity:.85;font-size:.92rem;margin:.15rem 0}
</style>
"""


def inject_report_css() -> None:
    st.markdown(REPORT_CSS, unsafe_allow_html=True)
