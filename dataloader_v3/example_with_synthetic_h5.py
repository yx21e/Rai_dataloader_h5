#!/usr/bin/env python3

from pathlib import Path
import sys
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader_v3 import GeoLoadInput, load_data, load_sample_h5, save_sample_h5, to_torch_batch
from dataloader.adapters import firms as firms_mod


def describe_sample(sample) -> None:
    x_shape = tuple(sample.x.shape)
    y_shape = tuple(sample.y.shape)
    channels = list(sample.meta.get("channels", []))

    print("load status: completed. reporting normalized output dimensions.")
    print("x shape:", x_shape)
    print("y shape:", y_shape)

    if len(x_shape) == 4:
        t, c, h, w = x_shape
        print(
            f"x dimensions: T={t} (time steps), C={c} (feature channels={channels}), "
            f"H={h}, W={w} (spatial grid)"
        )
    else:
        print("dimensions: x is expected to be 4D as (T, C, H, W)")

    if len(y_shape) == 3:
        yt, yh, yw = y_shape
        print(f"label grid: T={yt}, H={yh}, W={yw} (aligned with x in time and space)")

        cfg = sample.meta.get("config", {})
        label_cfg = cfg.get("label", {}) if isinstance(cfg, dict) else {}
        label_mapping = label_cfg.get("label_mapping", {}) if isinstance(label_cfg, dict) else {}
        selected = label_cfg.get("label_hazards", None) if isinstance(label_cfg, dict) else None
        default_id = label_cfg.get("label_default_value", 0) if isinstance(label_cfg, dict) else 0

        legend: list[str] = [f"{default_id}=default/background"]
        selected_set = set(str(h).strip().lower() for h in selected) if selected else set()
        y_codes = set(int(v) for v in np.unique(sample.y).tolist())

        for hazard, code in label_mapping.items():
            hz = str(hazard).strip().lower()
            cid = int(code)
            if hz in selected_set or cid in y_codes:
                legend.append(f"{cid}={hz}")

        seen = set()
        legend = [item for item in legend if not (item in seen or seen.add(item))]
        print("label legend:", ", ".join(legend))
    else:
        print("label grid: y is expected to be 3D as (T, H, W)")

    if len(x_shape) == 4 and len(y_shape) == 3:
        if x_shape[0] != y_shape[0]:
            print("warning: T mismatch between x and y")
        if x_shape[2] != y_shape[1] or x_shape[3] != y_shape[2]:
            print("warning: spatial mismatch between x and y")

    if len(x_shape) == 4 and x_shape[1] != len(channels):
        print(f"warning: channel count mismatch (C={x_shape[1]}, names={len(channels)})")

    if any(dim == 0 for dim in x_shape):
        print("note: x includes zero-sized dimension(s); no usable feature tensor for at least one axis")
    if any(dim == 0 for dim in y_shape):
        print("note: y includes zero-sized dimension(s); no usable label grid for at least one axis")


def main() -> None:
    firms_mod.FIRMSAdapter.CSV_PATTERNS = ["firmsFL14-25/*.csv"]
    firms_mod.FIRMSAdapter.JSON_PATTERNS = []

    req = GeoLoadInput(
        root_dir="/home/yangshuang",
        data_sources=["FIRMS"],
        temporal_window=("2023-01-01", "2023-01-02"),
        area_of_interest_bbox=(-87.8, 24.0, -79.8, 31.5),
        # spatial_resolution_deg=0.25,
        # synthetic_time=True,
        # temporal_cadence="D",
        # target_hazards=["wildfire"],
    )

    sample = load_data(req)
    describe_sample(sample)
    # print("x synthetic ratio:", float(sample.meta["x_synthetic_mask"].mean()))
    # print("y synthetic ratio:", float(sample.meta["y_synthetic_mask"].mean()))

    out_path = "/home/yangshuang/output/sample_v3.h5"
    save_sample_h5(sample, out_path)
    print(f"artifact status: HDF5 dataset persisted at {out_path}")

    sample2 = load_sample_h5(out_path)
    x_t, y_t, meta = to_torch_batch(sample2)
    print("torch tensors:", tuple(x_t.shape), tuple(y_t.shape), meta.get("channels", []))


if __name__ == "__main__":
    main()
