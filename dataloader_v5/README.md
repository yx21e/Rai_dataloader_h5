# Dataloader V5

V5 keeps the loading, HDF5, and visualization flow, and adds inspection/report utilities.

## API
```python
from dataloader_v5 import GeoLoadInput, load_data, inspect_sample, print_sample_summary, save_sample_report_html
```

## New in V5
- `inspect_sample(sample)` -> structured summary as a Python dict
- `print_sample_summary(sample)` -> human-readable terminal summary
- `save_sample_report_html(sample, path)` -> one-page HTML report

## Example
```python
from dataloader_v5 import GeoLoadInput, load_data, print_sample_summary, save_sample_report_html

req = GeoLoadInput(
    data_sources=["FIRMS", "NOAA", "ERA5"],
    temporal_window=("2023-01-01", "2023-01-10"),
    area_of_interest_bbox=(-87.8, 24.0, -79.8, 31.5),
    synthetic_time=True,
    temporal_cadence="D",
    target_hazards=["flood"],
    label_source="noaa",
)
sample = load_data(req)
print_sample_summary(sample)
save_sample_report_html(sample, "/home/yangshuang/output/sample_v5_report.html")
```
