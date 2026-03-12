# Rai Dataloader H5

Unified geospatial dataloader prototypes for multi-source hazard and environmental data.

This repository contains several iterations of the same pipeline:

- `dataloader/`: initial unified loader
- `dataloader_v2/`: synthetic time support and warning system
- `dataloader_v3/`: simplified request object, HDF5 export, PyTorch handoff
- `dataloader_v4/`: visualization and diagnostic utilities
- `dataloader_v5/`: reporting and publish-readiness checks

## Data policy

This repository contains code only. Local datasets are not included.

Expected local data roots are documented in the versioned READMEs and examples, for example:

- `firmsFL14-25/`
- `NOAA-NWMflood/`
- `era5/`
- `merra2/`
- `MTBS/`
- `USDALandfire/`

## Verified local status

The latest local verification was run with:

- `FIRMS`: load OK, synthetic OK, HDF5 round-trip OK
- `NOAA`: load OK, synthetic OK, HDF5 round-trip OK
- `ERA5`: load OK, synthetic OK, HDF5 round-trip OK
- `LANDFIRE`: load OK, synthetic OK, HDF5 round-trip OK
- `MTBS` as label source: load OK, synthetic alignment OK, HDF5 round-trip OK
- `MERRA2`: adapter exists, but not locally verified in the current workspace because source files were absent

## Quick start

Typical workflow:

```python
from dataloader_v3 import GeoLoadInput, load_data, save_sample_h5, load_sample_h5, to_torch_batch

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
save_sample_h5(sample, "sample.h5")
sample2 = load_sample_h5("sample.h5")
x_t, y_t, meta = to_torch_batch(sample2)
```

## Install

```bash
pip install -r requirements.txt
```
