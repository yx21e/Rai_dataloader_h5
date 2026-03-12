from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GridConfig:
    resolution_deg: float = 0.1
    bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass(frozen=True)
class TimeConfig:
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    frequency: str = "D"


@dataclass(frozen=True)
class LabelConfig:
    # where y comes from: mtbs | noaa | firms | None
    label_source: Optional[str] = "mtbs"
    # which hazards the client is interested in (canonical names)
    label_hazards: Optional[List[str]] = None
    # canonical hazard -> label id
    label_mapping: Dict[str, int] = field(
        default_factory=lambda: {
            "wildfire": 1,
            "flood": 2,
            "thunderstorm": 3,
            "tornado": 4,
            "marine": 5,
            "prescribed_fire": 6,
            "wildland_fire_use": 7,
            "unknown_fire": 8,
            "out_of_area_response": 9,
            "complex_fire": 10,
        }
    )
    label_default_value: int = 0
    # used when multiple hazards overlap on one time/cell (left to right = higher priority)
    label_priority: List[str] = field(
        default_factory=lambda: ["tornado", "flood", "thunderstorm", "marine", "wildfire"]
    )
    # FIRMS label quality controls (used when label_source="firms")
    firms_type_allowlist: Optional[List[str]] = None
    firms_min_confidence: Optional[float] = None


@dataclass(frozen=True)
class SyntheticConfig:
    synthetic_time: bool = False
    target_freq: str = "D"  # e.g., D, H, min
    continuous_method: str = "linear"  # linear|nearest|ffill
    event_method: str = "ffill"  # ffill|nearest|linear
    label_method: str = "nearest"  # nearest|ffill


@dataclass(frozen=True)
class PipelineV2Config:
    root_dir: str = "/home/yangshuang"
    grid: GridConfig = field(default_factory=GridConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    sources: List[str] = field(default_factory=lambda: ["firms", "mtbs_fod"])
    cache_dir: Optional[str] = None
