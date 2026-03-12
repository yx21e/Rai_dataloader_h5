from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from dataloader.config import GridConfig, LabelConfig, PipelineConfig, TimeConfig
from dataloader.dataset import UnifiedDataLoader
from dataloader.schema import Sample


def _normalize_sources(sources: Sequence[str]) -> List[str]:
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
    normalized = []
    for src in sources:
        key = str(src).strip().lower()
        if key not in mapping:
            raise KeyError(f"Unknown data source: {src}")
        normalized.append(mapping[key])
    return sorted(set(normalized))


def load_data(
    data: Sequence[str],
    date_range: Tuple[str, str],
    bbox: Tuple[float, float, float, float],
    resolution: float = 0.1,
    root_dir: str = "/home/yangshuang",
    label_map: Optional[Dict[str, int]] = None,
    label_rule: str = "max",
    cache_dir: Optional[str] = None,
) -> Sample:
    """Simple, explicit API for unified wildfire data loading."""
    sources = _normalize_sources(data)
    label_cfg = LabelConfig(
        incid_type_map=label_map or LabelConfig().incid_type_map,
        label_rule=label_rule,
    )
    cfg = PipelineConfig(
        root_dir=root_dir,
        grid=GridConfig(resolution_deg=resolution, bbox=bbox),
        time=TimeConfig(start_date=date_range[0], end_date=date_range[1]),
        label=label_cfg,
        sources=sources,
        cache_dir=cache_dir,
    )
    return UnifiedDataLoader(cfg).build()
