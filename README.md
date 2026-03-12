# Rai Dataloader H5

Unified geospatial dataloader for multi-source hazard and environmental data.

## What is included

- `dataloader/`: single maintained package
- `example_with_synthetic.py`: minimal synthetic-time example
- `example_no_synthetic.py`: missing-data warning example
- `example_with_synthetic_h5.py`: HDF5 export and PyTorch handoff
- `example_report.py`: structured sample summary
- `verify_publish_readiness.py`: per-source validation script

## Core API

```python
from dataloader import GeoLoadInput, load_data

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

`sample.x` has shape `(T, C, H, W)`, `sample.y` has shape `(T, H, W)`.

## HDF5 workflow

```python
from dataloader import load_sample_h5, save_sample_h5, to_torch_batch

save_sample_h5(sample, "sample.h5")
sample2 = load_sample_h5("sample.h5")
x_t, y_t, meta = to_torch_batch(sample2)
```

## Verified local support

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

## Data policy

This repository contains code only. Local datasets are not included.
