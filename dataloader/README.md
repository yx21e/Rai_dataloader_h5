# Unified Dataloader (Local Pipeline)

This module provides one main function:
- `load_data(...)` -> returns `Sample(x, y, meta)`

`x` is the feature tensor (`[T, C, H, W]`), `y` is the label tensor (`[T, H, W]`).
`y` is automatically aligned to `x` time coordinates (missing label dates are zero-filled).

## Example 1: Simple API (Main Function + Customizable Inputs)
```python
from dataloader import load_data

sample = load_data(
    # Choose any combination of sources:
    # "FIRMS", "MTBS", "ERA5", "NOAA", "LANDFIRE", "MERRA2"
    data=["FIRMS", "ERA5", "NOAA"],

    # Time range (inclusive)
    date_range=("2023-01-01", "2023-01-31"),

    # Spatial range: (min_lon, min_lat, max_lon, max_lat)
    bbox=(-87.8, 24.0, -79.8, 31.5),

    # Target grid resolution in degrees
    resolution=0.25,

    # Project root path
    root_dir="/home/yangshuang",

    # Optional cache for processed outputs
    cache_dir="/home/yangshuang/processed_cache",
)

print("x shape:", sample.x.shape)
print("y shape:", sample.y.shape)
print("channels:", sample.meta.get("channels"))
```


## Example 2: Pull Data with Wildfire Label Only
This example keeps only wildfire as positive label (`1`) and maps all other MTBS incident types to `0`.

```python
from dataloader import load_data

wildfire_only_map = {
    "unk": 0,
    "unknown": 0,
    "": 0,
    "wf": 1,
    "wildfire": 1,
    "rx": 0,
    "prescribed": 0,
    "prescribed fire": 0,
    "wildland fire use": 0,
    "out of area response": 0,
    "complex": 0,
}

sample = load_data(
    data=["FIRMS", "MTBS"],
    date_range=("2022-01-01", "2023-12-31"),
    bbox=(-87.8, 24.0, -79.8, 31.5),
    resolution=0.25,
    root_dir="/home/yangshuang",
    label_map=wildfire_only_map,
)

print("x shape:", sample.x.shape)
print("y shape:", sample.y.shape)
print("wildfire pixels:", int((sample.y == 1).sum()))
print("non-wildfire pixels:", int((sample.y == 0).sum()))
```
