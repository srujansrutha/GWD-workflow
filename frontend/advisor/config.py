"""Settings for the field advisor. Everything here can be overridden with environment variables."""
from __future__ import annotations

import os
from pathlib import Path

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("ADVISOR_MODEL", "qwen3.5:9b")
# 6144 keeps qwen3.5:9b fully on an 8 GB GPU next to the counter model; 8192 pushed about 12% of it onto the CPU.
OLLAMA_NUM_CTX = int(os.environ.get("ADVISOR_NUM_CTX", "6144"))
OLLAMA_KEEP_ALIVE = os.environ.get("ADVISOR_KEEP_ALIVE", "10m")
FORCE_FALLBACK = os.environ.get("ADVISOR_FORCE_FALLBACK") == "1"   # skip the LLM and use the rule-based text

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

USER_AGENT = "WheatFieldAdvisor/0.1 (local research tool)"
HTTP_TIMEOUT = 25

# --- counting defaults
CONF_DEFAULT = 0.25            # best count accuracy on our labelled photos (see docs/METRICS.md)
COUNTER_ERROR_UNCALIBRATED = 0.10   # extra relative uncertainty from the counter itself
COUNTER_ERROR_CALIBRATED = 0.05

# --- agronomy placeholders. These are generic values, NOT local recommendations: an agronomist
# should replace them per region and variety (see frontend/advisor/knowledge.py).
HEADS_REFERENCE = (400, 600)   # heads per m2 expected for a healthy stand at heading/grain fill
KERNELS_PER_HEAD = (28, 34, 40)    # low, central, high
TKW_G = (35, 42, 48)               # thousand-kernel weight, grams: low, central, high
CV_MODERATE, CV_HIGH = 15.0, 25.0  # photo-to-photo variation in percent

STAGES = ["Tillering", "Stem elongation", "Heading", "Flowering", "Grain fill", "Ripening"]
HEADS_VISIBLE_STAGES = {"Heading", "Flowering", "Grain fill", "Ripening"}
