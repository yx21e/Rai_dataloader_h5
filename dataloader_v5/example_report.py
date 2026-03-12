#!/usr/bin/env python3

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader.adapters import firms as firms_mod
from dataloader_v5 import GeoLoadInput, load_data, print_sample_summary, save_sample_report_html


def main() -> None:
    firms_mod.FIRMSAdapter.CSV_PATTERNS = ["firmsFL14-25/*.csv"]
    firms_mod.FIRMSAdapter.JSON_PATTERNS = []

    req = GeoLoadInput(
        root_dir="/home/yangshuang",
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
    print_sample_summary(sample)
    report_path = save_sample_report_html(sample, "/home/yangshuang/output/sample_v5_report.html")
    print("report saved:", report_path)


if __name__ == "__main__":
    main()
