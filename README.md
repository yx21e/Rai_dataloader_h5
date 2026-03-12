# Rai Dataloader H5

Unified geospatial dataloader for local multi-source hazard and environmental data.

## Main API

```python
from dataloader import GeoLoadInput, load_data
```

```python
req = GeoLoadInput(
    data_sources=["FIRMS", "NOAA", "ERA5"],
    temporal_window=("2023-01-01", "2023-01-10"),
    area_of_interest_bbox=(-87.8, 24.0, -79.8, 31.5),
    spatial_resolution_deg=0.25,
    synthetic_time=True,
    temporal_cadence="D",
    target_hazards=["flood"],
    label_source="noaa",
)

sample = load_data(req)
```

## `GeoLoadInput` Parameters

- `data_sources: list[str]`
  Supported values:
  - `"FIRMS"`
  - `"NOAA"`
  - `"ERA5"`
  - `"MTBS"`
  - `"LANDFIRE"`
  - `"MERRA2"`

- `temporal_window: tuple[str, str]`
  Format:
  - `("YYYY-MM-DD", "YYYY-MM-DD")`
  Example:
  - `("2023-01-01", "2023-01-10")`

- `area_of_interest_bbox: tuple[float, float, float, float]`
  Format:
  - `(min_lon, min_lat, max_lon, max_lat)`
  Example:
  - `(-87.8, 24.0, -79.8, 31.5)`

- `spatial_resolution_deg: float`
  Meaning:
  - output grid resolution in degrees
  Example:
  - `0.25`

- `root_dir: str = "/home/yangshuang"`
  Meaning:
  - local root directory that contains the raw datasets

- `synthetic_time: bool = False`
  Meaning:
  - `False`: only use observed timestamps
  - `True`: fill missing timestamps onto the requested cadence

- `temporal_cadence: str = "D"`
  Supported examples:
  - `"D"` for daily
  - `"12h"` for 12-hour
  - `"h"` for hourly
  - `"15min"` for 15-minute

- `target_hazards: list[str] | None = None`
  Supported values:
  - `"wildfire"`
  - `"flood"`
  - `"thunderstorm"`
  - `"tornado"`
  - `"marine"`
  - `"prescribed_fire"`
  - `"wildland_fire_use"`
  - `"unknown_fire"`
  - `"out_of_area_response"`
  - `"complex_fire"`

- `label_source: str | None = None`
  Supported values:
  - `"firms"`
  - `"noaa"`
  - `"mtbs"`
  - `None`
  Meaning:
  - if `None`, the loader tries to infer the label source from `data_sources` and `target_hazards`

- `label_mapping: dict[str, int] | None = None`
  Meaning:
  - optional explicit hazard-to-label-id mapping
  Default behavior:
  - if `target_hazards` is given and `label_mapping` is omitted, the loader uses binary encoding
  - example: `["wildfire"] -> {"wildfire": 1}`

## Output Format

`load_data(req)` returns `Sample(x, y, meta)`.

- `sample.x`
  Shape:
  - `(T, C, H, W)`
  Meaning:
  - `T`: time steps
  - `C`: feature channels
  - `H`: grid height
  - `W`: grid width

- `sample.y`
  Shape:
  - `(T, H, W)`
  Meaning:
  - label grid aligned with `x` in time and space

- `sample.meta`
  Common fields:
  - `channels`
  - `config`
  - `x_synthetic_mask`
  - `y_synthetic_mask`

## Source Behavior

- `FIRMS`
  Output:
  - `x` channels: `count`, `frp`
  Optional label source:
  - `label_source="firms"` for `wildfire`

- `NOAA`
  Output:
  - `x` channel: `noaa_flood`
  Optional label source:
  - `label_source="noaa"` for `flood`, `thunderstorm`, `tornado`, `marine`

- `ERA5`
  Output:
  - `x` channels: `t2m`, `d2m`, `u10`, `v10`, `sp`, `swvl1`

- `MTBS`
  Current use:
  - label source only
  Typical hazards:
  - `wildfire`
  - `prescribed_fire`
  - `wildland_fire_use`
  - `unknown_fire`
  - `out_of_area_response`
  - `complex_fire`

- `LANDFIRE`
  Output:
  - `x` channel: `landfire_fuel`
  Note:
  - static layer repeated across time if needed

- `MERRA2`
  Output:
  - continuous weather/environment channels when local files exist
  Current local status:
  - adapter exists, but local verification failed because source files were absent

## Default Synthetic Rules

When `synthetic_time=True`, the loader uses:

- continuous sources:
  - linear interpolation
- event sources:
  - forward fill
- labels:
  - nearest timestamp

## HDF5 Workflow

```python
from dataloader import load_sample_h5, save_sample_h5, to_torch_batch

save_sample_h5(sample, "sample.h5")
sample2 = load_sample_h5("sample.h5")
x_t, y_t, meta = to_torch_batch(sample2)
```

## Examples

- `example_with_synthetic.py`
  Minimal synthetic-time example

- `example_no_synthetic.py`
  Missing-data warning example

- `example_with_synthetic_h5.py`
  HDF5 export and PyTorch handoff

- `example_report.py`
  Structured sample summary

- `verify_publish_readiness.py`
  Per-source validation script

## Verified Local Support

- `FIRMS`: load OK, synthetic OK, HDF5 round-trip OK
- `NOAA`: load OK, synthetic OK, HDF5 round-trip OK
- `ERA5`: load OK, synthetic OK, HDF5 round-trip OK
- `LANDFIRE`: load OK, synthetic OK, HDF5 round-trip OK
- `MTBS` as label source: load OK, synthetic alignment OK, HDF5 round-trip OK
- `MERRA2`: adapter included, but local source files were absent during verification

## Install

```bash
pip install -r requirements.txt
```

## Data Policy

This repository contains code only. Local datasets are not included.
