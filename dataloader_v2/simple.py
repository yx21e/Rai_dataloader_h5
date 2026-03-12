from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, List

from dataloader.schema import Sample
from dataloader_v2.config import GridConfig, LabelConfig, PipelineV2Config, SyntheticConfig, TimeConfig
from dataloader_v2.dataset import UnifiedDataLoaderV2


def _normalize_sources(sources: Sequence[str]) -> list[str]:
    mapping = {
        "firms": "firms",
        "firm": "firms",
        "mtbs": "mtbs_fod",
        "mtbs_fod": "mtbs_fod",
        "fod": "mtbs_fod",
        "era5": "era5",
        "merra2": "merra2",
        "noaa": "noaa",
        "landfire": "landfire",
    }
    out: list[str] = []
    for src in sources:
        key = str(src).strip().lower()
        if key not in mapping:
            raise KeyError(f"Unknown data source: {src}")
        out.append(mapping[key])
    return sorted(set(out))


def load_data(
    data: Sequence[str],
    date_range: Tuple[str, str],
    bbox: Tuple[float, float, float, float],
    resolution: float = 0.1,
    root_dir: str = "/home/yangshuang",
    label_source: Optional[str] = "mtbs",
    label_hazards: Optional[List[str]] = None,
    label_mapping: Optional[Dict[str, int]] = None,
    label_default_value: int = 0,
    label_priority: Optional[List[str]] = None,
    firms_type_allowlist: Optional[List[str]] = None,
    firms_min_confidence: Optional[float] = None,
    cache_dir: Optional[str] = None,
    synthetic_time: bool = False,
    target_freq: str = "D",
    continuous_method: str = "linear",
    event_method: str = "ffill",
    label_method: str = "nearest",
) -> Sample:
    cfg = PipelineV2Config(
        root_dir=root_dir,
        grid=GridConfig(resolution_deg=resolution, bbox=bbox),
        time=TimeConfig(start_date=date_range[0], end_date=date_range[1], frequency=target_freq),
        label=LabelConfig(
            label_source=None if label_source is None else str(label_source).strip().lower(),
            label_hazards=[h.strip().lower() for h in label_hazards] if label_hazards else None,
            label_mapping=label_mapping or LabelConfig().label_mapping,
            label_default_value=label_default_value,
            label_priority=[h.strip().lower() for h in label_priority] if label_priority else LabelConfig().label_priority,
            firms_type_allowlist=[str(v).strip() for v in firms_type_allowlist] if firms_type_allowlist else None,
            firms_min_confidence=firms_min_confidence,
        ),
        synthetic=SyntheticConfig(
            synthetic_time=synthetic_time,
            target_freq=target_freq,
            continuous_method=continuous_method,
            event_method=event_method,
            label_method=label_method,
        ),
        sources=_normalize_sources(data),
        cache_dir=cache_dir,
    )
    return UnifiedDataLoaderV2(cfg).build()
