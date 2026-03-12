#!/usr/bin/env python3

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader_v3 import GeoLoadInput, load_data, save_sample_h5
from dataloader.adapters import firms as firms_mod


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
    out_path = save_sample_h5(sample, "/home/yangshuang/output/sample_v3.h5")
    print("saved h5:", out_path)
    print("x shape:", sample.x.shape)
    print("y shape:", sample.y.shape)


if __name__ == "__main__":
    main()
