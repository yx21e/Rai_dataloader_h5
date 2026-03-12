#!/usr/bin/env python3

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader.adapters import firms as firms_mod
from dataloader.registry import get_adapter
from dataloader_v2.config import TimeConfig
from dataloader_v4 import (
    GeoLoadInput,
    load_data,
    save_feature_label_png,
    save_feature_only_png,
    save_points_png,
    save_real_vs_synthetic_pairs_png,
)


def main() -> None:
    bbox = (-87.8, 24.0, -79.8, 31.5)
    outputs: list[str] = []

    firms_mod.FIRMSAdapter.CSV_PATTERNS = ["firmsFL14-25/*.csv"]
    firms_mod.FIRMSAdapter.JSON_PATTERNS = []

    # NOAA flood feature + label
    noaa_req = GeoLoadInput(
        root_dir="/home/yangshuang",
        data_sources=["NOAA"],
        temporal_window=("2023-01-01", "2023-01-05"),
        area_of_interest_bbox=bbox,
        spatial_resolution_deg=0.25,
        synthetic_time=False,
        target_hazards=["flood"],
        label_source="noaa",
    )
    noaa_sample = load_data(noaa_req)
    outputs.append(
        save_feature_label_png(
            noaa_sample,
            "/home/yangshuang/output/noaa_flood_feature_label.png",
            time_index=0,
            feature_name="noaa_flood",
        )
    )

    # FIRMS point map
    firms_df = get_adapter("firms")().load_points(
        "/home/yangshuang",
        TimeConfig(start_date="2023-01-01", end_date="2023-01-05", frequency="D"),
    )
    firms_df = firms_df[
        (firms_df["longitude"] >= bbox[0])
        & (firms_df["longitude"] <= bbox[2])
        & (firms_df["latitude"] >= bbox[1])
        & (firms_df["latitude"] <= bbox[3])
    ].copy()
    outputs.append(
        save_points_png(
            firms_df,
            "/home/yangshuang/output/firms_points_frp.png",
            bbox=bbox,
            color_column="frp",
            title="FIRMS detections colored by FRP",
        )
    )

    # ERA5 single-feature view
    era5_req = GeoLoadInput(
        root_dir="/home/yangshuang",
        data_sources=["ERA5"],
        temporal_window=("2023-01-01", "2023-01-02"),
        area_of_interest_bbox=bbox,
        spatial_resolution_deg=0.25,
        synthetic_time=False,
    )
    era5_sample = load_data(era5_req)
    outputs.append(
        save_feature_only_png(
            era5_sample,
            "/home/yangshuang/output/era5_t2m_feature.png",
            time_index=0,
            feature_name="t2m",
            title_prefix="ERA5 field",
        )
    )

    # LANDFIRE static feature view
    landfire_req = GeoLoadInput(
        root_dir="/home/yangshuang",
        data_sources=["LANDFIRE"],
        temporal_window=("2023-01-01", "2023-01-02"),
        area_of_interest_bbox=bbox,
        spatial_resolution_deg=0.25,
        synthetic_time=False,
    )
    landfire_sample = load_data(landfire_req)
    outputs.append(
        save_feature_only_png(
            landfire_sample,
            "/home/yangshuang/output/landfire_fuel_feature.png",
            time_index=0,
            feature_name="landfire_fuel",
            title_prefix="LANDFIRE field",
        )
    )

    # NOAA real vs synthetic pairs
    noaa_syn_req = GeoLoadInput(
        root_dir="/home/yangshuang",
        data_sources=["NOAA"],
        temporal_window=("2023-01-01", "2023-01-15"),
        area_of_interest_bbox=bbox,
        spatial_resolution_deg=0.25,
        synthetic_time=True,
        temporal_cadence="D",
        target_hazards=["flood"],
        label_source="noaa",
    )
    noaa_syn_sample = load_data(noaa_syn_req)
    outputs.append(
        save_real_vs_synthetic_pairs_png(
            noaa_syn_sample,
            "/home/yangshuang/output/noaa_real_vs_synthetic.png",
            feature_name="noaa_flood",
            max_pairs=1,
        )
    )

    print("generated visualizations:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
