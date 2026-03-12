from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class GridConfig:
    """Spatial grid configuration in EPSG:4326."""

    resolution_deg: float = 0.1
    bbox: Optional[Tuple[float, float, float, float]] = None  # minx, miny, maxx, maxy


@dataclass(frozen=True)
class TimeConfig:
    """Time configuration. Dates are inclusive and in YYYY-MM-DD."""

    start_date: Optional[str] = None
    end_date: Optional[str] = None
    frequency: str = "D"


@dataclass(frozen=True)
class LabelConfig:
    """Label configuration for MTBS FOD incident type."""

    incid_type_map: Dict[str, int] = field(
        default_factory=lambda: {
            "unk": 0,
            "unknown": 0,
            "": 0,
            "wf": 1,
            "wildfire": 1,
            "rx": 2,
            "prescribed": 2,
            "prescribed fire": 2,
            "wildland fire use": 3,
            "out of area response": 4,
            "complex": 5,
        }
    )
    label_rule: str = "max"


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level pipeline configuration."""

    root_dir: str = "/home/yangshuang"
    grid: GridConfig = field(default_factory=GridConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    label: LabelConfig = field(default_factory=LabelConfig)
    sources: List[str] = field(default_factory=lambda: ["firms", "mtbs_fod"])
    cache_dir: Optional[str] = None
