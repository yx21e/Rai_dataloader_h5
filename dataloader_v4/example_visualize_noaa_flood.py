#!/usr/bin/env python3

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader_v4 import GeoLoadInput, load_data, save_feature_label_png


def main() -> None:
    req = GeoLoadInput(
        root_dir="/home/yangshuang",
        data_sources=["NOAA"],
        temporal_window=("2023-01-01", "2023-01-05"),
        area_of_interest_bbox=(-87.8, 24.0, -79.8, 31.5),
        spatial_resolution_deg=0.25,
        synthetic_time=False,
        target_hazards=["flood"],
        label_source="noaa",
    )
    sample = load_data(req)
    out = save_feature_label_png(
        sample,
        output_path="/home/yangshuang/output/noaa_flood_feature_label.png",
        time_index=0,
        feature_name="noaa_flood",
    )
    print("visualization saved:", out)


if __name__ == "__main__":
    main()
