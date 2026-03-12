#!/usr/bin/env python3

import warnings

from dataloader import GeoLoadInput, load_data
from dataloader.adapters import firms as firms_mod


def main() -> None:
    warnings.simplefilter("always", UserWarning)

    firms_mod.FIRMSAdapter.CSV_PATTERNS = ["firmsFL14-25/*.csv"]
    firms_mod.FIRMSAdapter.JSON_PATTERNS = []

    request = GeoLoadInput(
        root_dir="/home/yangshuang",
        data_sources=["FIRMS"],
        temporal_window=("1905-01-01", "1905-01-02"),
        area_of_interest_bbox=(-87.8, 24.0, -79.8, 31.5),
        spatial_resolution_deg=0.25,
        synthetic_time=False,
        temporal_cadence="D",
        target_hazards=["wildfire"],
    )
    sample = load_data(request)

    print("load status: completed. reporting normalized output dimensions.")
    print("x shape:", tuple(sample.x.shape))
    print("y shape:", tuple(sample.y.shape))
    print("channels:", list(sample.meta.get("channels", [])))


if __name__ == "__main__":
    main()
