"""Typed inputs and the structured report the language model must produce."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

LEVELS = ["Unknown", "Low", "Medium", "High"]
TEXTURES = ["Unknown", "Sand", "Loamy sand", "Sandy loam", "Loam", "Silt loam", "Silt", "Sandy clay loam",
            "Clay loam", "Silty clay loam", "Sandy clay", "Silty clay", "Clay"]


class SoilInput(BaseModel):
    """Soil test results. P, K and N are the lab's own rating, so units and methods never get misread."""
    ph: Optional[float] = Field(None, ge=3.0, le=10.5)
    organic_matter_pct: Optional[float] = Field(None, ge=0, le=30)
    texture: str = "Unknown"
    p_level: str = "Unknown"
    k_level: str = "Unknown"
    n_level: str = "Unknown"
    test_date: Optional[date] = None

    @property
    def provided(self) -> bool:
        return any([self.ph is not None, self.organic_matter_pct is not None, self.texture != "Unknown",
                    self.p_level != "Unknown", self.k_level != "Unknown", self.n_level != "Unknown"])


class FieldInput(BaseModel):
    name: str = "My field"
    crop: str = "Wheat"
    variety: str = ""
    sowing_date: date
    growth_stage: str
    irrigated: bool = False
    previous_crop: str = ""
    inputs_applied: str = ""        # fertilizer / sprays so far, free text
    problems_noticed: str = ""
    target_yield_t_ha: Optional[float] = Field(None, ge=0, le=20)

    lat: Optional[float] = Field(None, ge=-90, le=90)
    lon: Optional[float] = Field(None, ge=-180, le=180)
    polygon: list[tuple[float, float]] = Field(default_factory=list)   # (lat, lon) corners of the field
    area_ha: Optional[float] = Field(None, ge=0)

    photo_area_m2: Optional[float] = Field(None, gt=0, le=1000)        # ground area visible in each photo
    kernels_per_head: Optional[float] = Field(None, gt=0, le=100)
    tkw_g: Optional[float] = Field(None, gt=0, le=100)
    reference_heads_low: Optional[float] = Field(None, ge=0)
    reference_heads_high: Optional[float] = Field(None, ge=0)
    conf: float = Field(0.25, ge=0.10, le=0.90)

    soil: SoilInput = Field(default_factory=SoilInput)


@dataclass
class PhotoIn:
    name: str
    raw: bytes
    hand_count: Optional[int] = None


# ---------------------------------------------------------------- what the language model must return
class Reason(BaseModel):
    text: str
    evidence_ids: list[str]


class ActionNote(BaseModel):
    action_id: str
    explanation: str
    evidence_ids: list[str]


class ReportDraft(BaseModel):
    summary: str
    reasons: list[Reason]
    actions: list[ActionNote]
    open_questions: list[str]
    limitations: str


@dataclass
class RunStats:
    attempts: int = 0
    used_fallback: bool = False
    model: str = ""
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
