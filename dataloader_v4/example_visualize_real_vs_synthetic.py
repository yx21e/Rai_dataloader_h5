#!/usr/bin/env python3

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader_v4 import (
    GeoLoadInput,
    infer_observed_and_synthetic_pairs,
    load_data,
    save_real_vs_synthetic_pairs_png,
)
from dataloader_v4.visualize import _time_label


def main() -> None:
    req = GeoLoadInput(
        root_dir="/home/yangshuang",
        data_sources=["NOAA"],
        temporal_window=("2023-01-01", "2023-01-15"),
        area_of_interest_bbox=(-87.8, 24.0, -79.8, 31.5),
        spatial_resolution_deg=0.25,
        synthetic_time=True,
        temporal_cadence="D",
        target_hazards=["flood"],
        label_source="noaa",
    )
    sample = load_data(req)
    pairs = infer_observed_and_synthetic_pairs(sample, feature_name="noaa_flood", max_pairs=3)
    out = save_real_vs_synthetic_pairs_png(
        sample,
        output_path="/home/yangshuang/output/noaa_real_vs_synthetic.png",
        feature_name="noaa_flood",
        max_pairs=3,
    )
    for idx, (prev_obs_idx, synthetic_idx, next_obs_idx) in enumerate(pairs, start=1):
        print(
            f"pair {idx}: previous_observed={prev_obs_idx} {_time_label(sample, prev_obs_idx)} | "
            f"synthetic={synthetic_idx} {_time_label(sample, synthetic_idx)} | "
            f"next_observed={next_obs_idx} {_time_label(sample, next_obs_idx)}"
        )
    print("visualization saved:", out)


if __name__ == "__main__":
    main()
