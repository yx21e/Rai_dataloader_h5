# Dataloader V4

V4 keeps the V3 loading/export flow and adds a visualization utility.

## API
```python
from dataloader_v4 import GeoLoadInput, load_data, save_sample_overview_png
```

## Visualization
`save_sample_overview_png(sample, output_path, time_index=0, channel_index=0, feature_name=None)`

Outputs a PNG with:
- one `x` feature map
- `y` label grid
- `x` synthetic mask
- `y` synthetic mask

Additional visualization utilities:
- `save_feature_label_png(...)`: best for hazard/event grids such as NOAA flood
- `save_points_png(...)`: best for point datasets such as FIRMS detections
- `save_real_vs_synthetic_png(...)`: compare observed vs synthetic slices for one feature

## Example
```python
from dataloader_v4 import GeoLoadInput, load_data, save_sample_overview_png

req = GeoLoadInput(
    data_sources=["FIRMS"],
    temporal_window=("2023-01-01", "2023-01-02"),
    area_of_interest_bbox=(-87.8, 24.0, -79.8, 31.5),
    synthetic_time=True,
    target_hazards=["wildfire"],
)
sample = load_data(req)
save_sample_overview_png(sample, "/home/yangshuang/output/sample_v4_overview.png", feature_name="frp")
```
