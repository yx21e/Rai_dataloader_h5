#!/usr/bin/env python3

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader.adapters import firms as firms_mod
from dataloader_v4 import GeoLoadInput, load_data, save_sample_overview_png


def main() -> None:
    firms_mod.FIRMSAdapter.CSV_PATTERNS = ["firmsFL14-25/*.csv"]
    firms_mod.FIRMSAdapter.JSON_PATTERNS = []

    req = GeoLoadInput(
        root_dir="/home/yangshuang",
        data_sources=["FIRMS"],
        temporal_window=("2023-01-01", "2023-01-02"),
        area_of_interest_bbox=(-87.8, 24.0, -79.8, 31.5),
        spatial_resolution_deg=0.25,
        synthetic_time=True,
        temporal_cadence="D",
        target_hazards=["wildfire"],
    )
    sample = load_data(req)
    out = save_sample_overview_png(
        sample,
        output_path="/home/yangshuang/output/sample_v4_overview.png",
        time_index=0,
        feature_name="frp",
    )
    print("visualization saved:", out)


if __name__ == "__main__":
    main()
