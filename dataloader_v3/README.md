# Dataloader V3

Minimal API with a structured request object.

## API
```python
from dataloader_v3 import (
    GeoLoadInput,
    load_data,
    save_sample_h5,
    load_sample_h5,
    to_torch_batch,
)
```

## `GeoLoadInput` (required format)
- `data_sources: list[str]`  
  Supported sources: `FIRMS`, `ERA5`, `NOAA`, `MTBS`, `LANDFIRE`, `MERRA2`  
  Example: `["FIRMS", "ERA5", "NOAA", "MTBS", "LANDFIRE", "MERRA2"]`
- `temporal_window: tuple[str, str]`  
  Format: `("YYYY-MM-DD", "YYYY-MM-DD")` or `("YYYY-MM-DD HH:MM:SS", "...")`
- `area_of_interest_bbox: tuple[float, float, float, float]`  
  Order: `(min_lon, min_lat, max_lon, max_lat)`

## Optional
- `spatial_resolution_deg: float = 0.1`
- `root_dir: str = "/home/yangshuang"`
- `synthetic_time: bool = False`
- `temporal_cadence: str = "D"` (`"D"`, `"H"`, `"15min"`, ...)
- `target_hazards: list[str] | None = None`
  - Supported hazards (current): `wildfire`, `flood`, `thunderstorm`, `tornado`, `marine`,
    `prescribed_fire`, `wildland_fire_use`, `unknown_fire`, `out_of_area_response`, `complex_fire`
  - If set and no mapping is provided, default code is `1` for each selected hazard
- `label_source: str | None = None` (`"firms" | "noaa" | "mtbs"`, else auto-infer)
- `label_mapping: dict[str, int] | None = None`

## Example
```python
from dataloader_v3 import GeoLoadInput, load_data, save_sample_h5

req = GeoLoadInput(
    data_sources=["FIRMS", "ERA5", "NOAA", "MTBS", "LANDFIRE", "MERRA2"],
    temporal_window=("2023-01-01", "2023-01-02"),
    area_of_interest_bbox=(-87.8, 24.0, -79.8, 31.5),
    spatial_resolution_deg=0.25,
    synthetic_time=True,
    temporal_cadence="D",
    target_hazards=["wildfire"],
)
sample = load_data(req)
save_sample_h5(sample, "/home/yangshuang/output/sample_v3.h5")
sample2 = load_sample_h5("/home/yangshuang/output/sample_v3.h5")
x_t, y_t, meta = to_torch_batch(sample2)
```

## Multi-hazard Example
```python
req = GeoLoadInput(
    data_sources=["FIRMS", "NOAA"],
    temporal_window=("2023-01-01", "2023-01-31"),
    area_of_interest_bbox=(-87.8, 24.0, -79.8, 31.5),
    synthetic_time=True,
    temporal_cadence="D",
    target_hazards=["wildfire", "flood", "thunderstorm", "tornado"],
    label_mapping={"wildfire": 1, "flood": 2, "thunderstorm": 3, "tornado": 4},
)
```
