# Dataloader Package

Single maintained interface for unified geospatial loading, synthetic time filling,
HDF5 export, reporting, and visualization.

## Main entry point

```python
from dataloader import GeoLoadInput, load_data
```

## Input request

- `data_sources: list[str]`
- `temporal_window: (start, end)`
- `area_of_interest_bbox: (min_lon, min_lat, max_lon, max_lat)`
- `spatial_resolution_deg: float`
- `synthetic_time: bool`
- `temporal_cadence: str`
- `target_hazards: list[str] | None`
- `label_source: "firms" | "noaa" | "mtbs" | None`

## Output

- `sample.x`: `(T, C, H, W)`
- `sample.y`: `(T, H, W)`
- `sample.meta`: channels, config, synthetic masks

## Extra utilities

- `save_sample_h5`, `load_sample_h5`, `to_torch_batch`
- `inspect_sample`, `print_sample_summary`, `save_sample_report_html`
- visualization helpers in `dataloader.visualize`
