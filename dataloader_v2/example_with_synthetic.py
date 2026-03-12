#!/usr/bin/env python3

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader_v2 import load_data
from dataloader.adapters import firms as firms_mod


def main() -> None:
    # Demo-speed mode: avoid loading very large FIRMS JSON archives.
    firms_mod.FIRMSAdapter.CSV_PATTERNS = ["firmsFL14-25/*.csv"]
    firms_mod.FIRMSAdapter.JSON_PATTERNS = []

    sample = load_data(
        data=["FIRMS"],
        date_range=("2023-01-01", "2023-01-02"),
        bbox=(-87.8, 24.0, -79.8, 31.5),
        resolution=0.25,
        root_dir="/home/yangshuang",
        synthetic_time=True,
        target_freq="D",
        continuous_method="linear",
        event_method="ffill",
        label_method="nearest",
        label_source="firms",
        label_hazards=["wildfire"],
        label_mapping={"wildfire": 1},
    )
    print("x shape:", sample.x.shape)
    print("y shape:", sample.y.shape)
    print("channels:", sample.meta.get("channels"))
    print("x synthetic ratio:", float(sample.meta["x_synthetic_mask"].mean()))
    print("y synthetic ratio:", float(sample.meta["y_synthetic_mask"].mean()))


if __name__ == "__main__":
    main()
