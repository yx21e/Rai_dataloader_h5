from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from dataloader.config import GridConfig, LabelConfig, PipelineConfig, SyntheticConfig, TimeConfig
from dataloader.dataset import UnifiedDataLoader
from dataloader.schema import Sample


def _normalize_source_name(source: str) -> str:
    mapping = {
        "firms": "firms",
        "firm": "firms",
        "noaa": "noaa",
        "mtbs": "mtbs",
        "mtbs_fod": "mtbs",
        "fod": "mtbs",
        "era5": "era5",
        "merra2": "merra2",
        "landfire": "landfire",
    }
    return mapping.get(str(source).strip().lower(), str(source).strip().lower())


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


def _infer_label_source(
    data_sources: Sequence[str], target_hazards: Optional[Sequence[str]] = None
) -> Optional[str]:
    normalized = {_normalize_source_name(d) for d in data_sources}
    selected_hazards = {str(h).strip().lower() for h in target_hazards or []}

    noaa_hazards = {"flood", "thunderstorm", "tornado", "marine"}
    mtbs_hazards = {
        "wildfire",
        "prescribed_fire",
        "wildland_fire_use",
        "unknown_fire",
        "out_of_area_response",
        "complex_fire",
    }
    firms_hazards = {"wildfire"}

    if selected_hazards & noaa_hazards and "noaa" in normalized:
        return "noaa"
    if selected_hazards & (mtbs_hazards - firms_hazards) and "mtbs" in normalized:
        return "mtbs"
    if selected_hazards & firms_hazards and "firms" in normalized:
        return "firms"
    if selected_hazards & mtbs_hazards and "mtbs" in normalized:
        return "mtbs"

    for candidate in ("firms", "noaa", "mtbs"):
        if candidate in normalized:
            return candidate
    return None


def _default_binary_mapping(target_hazards: Optional[List[str]]) -> Optional[Dict[str, int]]:
    if not target_hazards:
        return None
    return {str(h).strip().lower(): 1 for h in target_hazards}


@dataclass(init=False)
class GeoLoadInput:
    r"""Structured request input of :func:`dataloader.load_data`.

    Args:
        data_sources (Sequence[str]): Input datasets to load and fuse.
        temporal_window (Tuple[str, str]): Query time window as ``(start, end)``.
        area_of_interest_bbox (Tuple[float, float, float, float]): AOI bounds
            as ``(min_lon, min_lat, max_lon, max_lat)``.
        spatial_resolution_deg (float, optional): Target grid resolution in
            degrees. (default: ``0.1``)
        root_dir (str, optional): Root folder of local datasets.
            (default: ``"/home/yangshuang"``)
        synthetic_time (bool, optional): Enable temporal harmonization and
            gap filling. (default: ``False``)
        temporal_cadence (str, optional): Target cadence such as ``"D"``,
            ``"h"``, ``"15min"``. (default: ``"D"``)
        target_hazards (List[str], optional): Hazards to encode in ``y``.
            If provided without ``label_mapping``, defaults to binary encoding
            (hazard -> ``1``). (default: ``None``)
        label_source (str, optional): Explicit label source override
            (``"firms"``, ``"noaa"``, ``"mtbs"``). If omitted, inferred from
            ``data_sources`` and ``target_hazards``. (default: ``None``)
        label_mapping (Dict[str, int], optional): Explicit hazard-to-id mapping.
            (default: ``None``)
    """

    data_sources: Sequence[str]
    temporal_window: Tuple[str, str]
    area_of_interest_bbox: Tuple[float, float, float, float]
    spatial_resolution_deg: float
    root_dir: str
    synthetic_time: bool
    temporal_cadence: str
    target_hazards: Optional[List[str]]
    label_source: Optional[str]
    label_mapping: Optional[Dict[str, int]]

    def __init__(
        self,
        data_sources: Sequence[str],
        temporal_window: Tuple[str, str],
        area_of_interest_bbox: Tuple[float, float, float, float],
        spatial_resolution_deg: float = 0.1,
        root_dir: str = "/home/yangshuang",
        synthetic_time: bool = False,
        temporal_cadence: str = "D",
        target_hazards: Optional[List[str]] = None,
        label_source: Optional[str] = None,
        label_mapping: Optional[Dict[str, int]] = None,
    ) -> None:
        self.data_sources = data_sources
        self.temporal_window = temporal_window
        self.area_of_interest_bbox = area_of_interest_bbox
        self.spatial_resolution_deg = spatial_resolution_deg
        self.root_dir = root_dir
        self.synthetic_time = synthetic_time
        self.temporal_cadence = temporal_cadence
        self.target_hazards = target_hazards
        self.label_source = label_source
        self.label_mapping = label_mapping


def load_data(request: GeoLoadInput) -> Sample:
    """Load one unified geospatial sample."""
    resolved_label_source = (
        str(request.label_source).strip().lower()
        if request.label_source is not None
        else _infer_label_source(request.data_sources, request.target_hazards)
    )
    resolved_mapping = request.label_mapping or _default_binary_mapping(request.target_hazards)

    cfg = PipelineConfig(
        root_dir=request.root_dir,
        grid=GridConfig(
            resolution_deg=request.spatial_resolution_deg,
            bbox=request.area_of_interest_bbox,
        ),
        time=TimeConfig(
            start_date=request.temporal_window[0],
            end_date=request.temporal_window[1],
            frequency=request.temporal_cadence,
        ),
        label=LabelConfig(
            label_source=resolved_label_source,
            label_hazards=[h.strip().lower() for h in request.target_hazards]
            if request.target_hazards
            else None,
            label_mapping=resolved_mapping or LabelConfig().label_mapping,
        ),
        synthetic=SyntheticConfig(
            synthetic_time=request.synthetic_time,
            target_freq=request.temporal_cadence,
            continuous_method="linear",
            event_method="ffill",
            label_method="nearest",
        ),
        sources=_normalize_sources(request.data_sources),
    )
    return UnifiedDataLoader(cfg).build()


def load_data_legacy(
    data: Sequence[str],
    date_range: Tuple[str, str],
    bbox: Tuple[float, float, float, float],
    resolution: float = 0.1,
    root_dir: str = "/home/yangshuang",
    synthetic_time: bool = False,
    target_freq: str = "D",
    label_hazards: Optional[List[str]] = None,
    label_source: Optional[str] = None,
    label_mapping: Optional[Dict[str, int]] = None,
) -> Sample:
    req = GeoLoadInput(
        data_sources=data,
        temporal_window=date_range,
        area_of_interest_bbox=bbox,
        spatial_resolution_deg=resolution,
        root_dir=root_dir,
        synthetic_time=synthetic_time,
        temporal_cadence=target_freq,
        target_hazards=label_hazards,
        label_source=label_source,
        label_mapping=label_mapping,
    )
    return load_data(req)
