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
# Beyond these, the input is almost certainly wrong (for example a mistyped photo area), so no verdict is given.
MAX_PLAUSIBLE_DENSITY = 1500       # heads per m2
MAX_PLAUSIBLE_YIELD = 14.0         # t/ha

# --- weather forecast: warnings and timing only. It never changes the verdict, the confidence or the yield range.
FORECAST_DAYS = 7                  # days fetched and shown to the reader
FORECAST_ACTION_DAYS = 5           # only the first days drive advice; forecasts get weaker after about 5 days
FORECAST_CACHE_HOURS = 3           # a saved forecast is reused for this long
FORECAST_STALE_HOURS = 12          # the report page warns when its forecast is older than this
FORECAST_RAIN_DAY_MM = 1.0         # a rainy day
FORECAST_NOTABLE_MM = 5.0          # the wettest day gets its own fact only when it is at least this wet
FORECAST_DRY_MM = 3.0              # less than this over the action window means no useful rain
FORECAST_HEAVY_RAIN_MM = 10.0      # enough rain to change irrigation or harvest plans
FORECAST_MIN_CHANCE = 50           # percent chance the service must give for that rain, when it gives one
FORECAST_HOT_C = 30.0
FORECAST_HOT_DAYS = 2              # hot days in the action window that trigger a heat warning
FORECAST_WET_DAYS = 3              # warm, humid, rainy days in the action window that raise disease risk
FORECAST_WET_MIN_TMAX = 15.0
FORECAST_WET_MIN_RH = 70.0

STAGES = ["Tillering", "Stem elongation", "Heading", "Flowering", "Grain fill", "Ripening"]
HEADS_VISIBLE_STAGES = {"Heading", "Flowering", "Grain fill", "Ripening"}
