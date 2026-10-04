"""General agronomy notes used to frame advice by growth stage.

These are broad, widely accepted statements. They are NOT local recommendations: thresholds, timings and
product rules differ by region and variety, and an agronomist should review and replace them.
"""
from __future__ import annotations

STAGE_NOTES = {
    "Tillering": "Head number is still being decided. Heads are not visible yet, so a head count says little.",
    "Stem elongation": "Head number is still being decided. Heads are not visible yet, so a head count says little.",
    "Heading": ("Heads are emerging and head number is already set. Late nitrogen rarely changes it. "
                "Disease protection decisions depend on region and the product label."),
    "Flowering": ("Flowering is the stage most sensitive to heat and frost. Warm, wet weather now raises the risk "
                  "of head diseases such as fusarium head blight."),
    "Grain fill": ("Yield now depends mostly on grain weight. Water stress and heat shorten grain fill. Irrigation, "
                   "where available, still helps. Leaf disease protects or reduces the grain-filling area."),
    "Ripening": ("Nothing applied now changes yield. The useful decisions are harvest timing, grain moisture, "
                 "storage and logistics."),
}

GENERAL_NOTES = [
    "A head count measures one yield component. Yield also depends on kernels per head and kernel weight.",
    "Most nitrogen and seeding decisions are made before heads appear, so photos mainly guide the forecast and next season.",
    "Soil pH around 6 to 7 suits most wheat. Strongly acid soils usually need liming; alkaline soils can limit some micronutrients.",
    "Do not give pesticide products or application doses. A licensed local advisor decides those.",
]


def stage_note(stage: str) -> str:
    return STAGE_NOTES.get(stage, "")
