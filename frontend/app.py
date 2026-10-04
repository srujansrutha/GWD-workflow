"""Wheat Advisor - two pages sharing one GPU model.

Run (from this folder):  streamlit run app.py        or double-click run_app.bat

  Head counter : upload photos, see detected wheat heads and counts.
  Field report : photos + field details in, a grounded condition report out (LangGraph + local Ollama model).

The detector is loaded onto the GPU once and stays there until the Streamlit process is stopped.
"""
import streamlit as st

st.set_page_config(page_title="Wheat Advisor", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

st.navigation([
    st.Page("counter_page.py", title="Head counter", icon="🌾", default=True),
    st.Page("report_page.py", title="Field report", icon="📋", url_path="report"),
]).run()
