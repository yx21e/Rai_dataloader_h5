# Dataloader V2

V2 adds optional synthetic temporal expansion and strict warning behavior for missing sources.

## Main API
```python
from dataloader_v2 import load_data
```

## Key Options
- Label controls:
  - `label_source`: `\"mtbs\" | \"noaa\" | \"firms\" | None`
  - `label_hazards`: hazards to keep as labels, e.g. `["wildfire"]`, `["flood","thunderstorm"]`
  - `label_mapping`: hazard-to-id map, e.g. `{"wildfire":1, "flood":2, "thunderstorm":3}`
  - `label_default_value`: background/default class id (usually `0`)
  - `label_priority`: priority when multiple hazards overlap in one time/cell
  - `firms_type_allowlist`: optional FIRMS `type` filter when `label_source="firms"`
  - `firms_min_confidence`: optional numeric confidence threshold when `label_source="firms"`
- `synthetic_time=False`:
  - No synthetic expansion.
  - If requested data source is missing/empty, a warning is emitted immediately:
    - `"Requested data source '<source>' is missing or has no data ..."`
- `synthetic_time=True`:
  - Expand to `target_freq` timeline and apply interpolation/approximation.
  - Per-type methods:
    - `continuous_method`: for continuous channels (`linear|nearest|ffill`)
    - `event_method`: for event channels (`ffill|nearest|linear`)
    - `label_method`: for labels (`nearest|ffill`)

## Source-to-Output Mapping (What Goes to `x` vs `y`)
This section defines exactly how each source is used.

- `FIRMS` -> **x features**
  - channels: `count`, `frp`
- `ERA5` -> **x features**
  - channels (current default): `t2m`, `d2m`, `u10`, `v10`, `sp`, `swvl1`
- `NOAA` -> **x features**
  - channel: `noaa_flood`
- `MERRA2` -> **x features**
  - channels depend on available variables in local files (if present)
- `LANDFIRE` -> **x features**
  - channel: `landfire_fuel` (static feature repeated over time)
- `MTBS` (`mtbs_fod`) -> **y labels**
  - `y` comes from `Incid_Type` after hazard mapping
- `NOAA` -> can also be **y labels**
  - based on NOAA phenomena -> hazard mapping (e.g. flood/thunderstorm/tornado)
- `FIRMS` -> can also be **y labels**
  - wildfire presence label from FIRMS detections (binary/mapped id)

Important behavior:
- `x` always has shape `(T, C, H, W)`.
- `y` always has shape `(T, H, W)`.
- `y` is controlled by `label_source`, not by `data` list membership.
- `sample.meta["channels"]` lists the exact order of channels in `x`.

## Why Shapes Still Exist When Data Is Missing
When requested feature sources are unavailable, V2 still returns tensors with consistent dimensions:
- `x` keeps temporal and spatial dimensions but has zero feature channels:
  - `x.shape = (T, 0, H, W)`
- `y` still follows the same temporal/spatial grid:
  - `y.shape = (T, H, W)`

Reason:
- This keeps the API contract stable for downstream code (`x/y/meta` always exist).
- It allows clients to detect missing features explicitly via `C=0` without crashing the pipeline.

## Observed vs Synthetic Flags
V2 explicitly exposes synthetic indicators in `sample.meta`:
- `sample.meta["x_synthetic_mask"]` -> shape `(T, C)`, `True` means synthetic at that time/channel.
- `sample.meta["y_synthetic_mask"]` -> shape `(T, H, W)`, `True` means synthetic label at that time/cell.

This lets clients distinguish real observed values from synthetic values.

## Example 1: Strict Mode (No Synthetic)
```python
from dataloader_v2 import load_data

sample = load_data(
    data=["FIRMS", "ERA5", "NOAA", "MERRA2"],  # MERRA2 may be missing locally
    date_range=("2023-01-01", "2023-01-31"),
    bbox=(-87.8, 24.0, -79.8, 31.5),
    resolution=0.25,
    root_dir="/home/yangshuang",
    synthetic_time=False,
    label_source="firms",
    label_hazards=["wildfire"],
    label_mapping={"wildfire": 1},
    firms_type_allowlist=["0"],  # optional
    firms_min_confidence=50,     # optional
)
print(sample.x.shape, sample.y.shape)
```

## Example 2: Synthetic Temporal Expansion
```python
from dataloader_v2 import load_data

sample = load_data(
    data=["FIRMS", "ERA5", "NOAA", "MTBS"],
    date_range=("2023-01-01", "2023-01-31"),
    bbox=(-87.8, 24.0, -79.8, 31.5),
    resolution=0.25,
    root_dir="/home/yangshuang",
    synthetic_time=True,
    target_freq="D",
    continuous_method="linear",
    event_method="ffill",
    label_method="nearest",
    label_source="noaa",
    label_hazards=["flood", "thunderstorm", "tornado"],
    label_mapping={"flood": 2, "thunderstorm": 3, "tornado": 4},
    label_priority=["tornado", "flood", "thunderstorm"],
)
print(sample.x.shape, sample.y.shape, sample.meta["channels"])
print("x synthetic ratio:", sample.meta["x_synthetic_mask"].mean())
print("y synthetic ratio:", sample.meta["y_synthetic_mask"].mean())
```
