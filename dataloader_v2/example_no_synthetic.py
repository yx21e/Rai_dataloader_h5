#!/usr/bin/env python3
# Should return warnings quickly.
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader_v2 import load_data


def main() -> None:
    sample = load_data(
        data=["FIRMS"],
        date_range=("1905-01-01", "1905-01-31"),
        bbox=(-87.8, 24.0, -79.8, 31.5),
        resolution=0.25,
        root_dir="/home/yangshuang",
        synthetic_time=False,
        label_source="firms",
        label_hazards=["wildfire"],
        label_mapping={"wildfire": 1},
    )
    print("x shape:", sample.x.shape)
    print("y shape:", sample.y.shape)
    print("channels:", sample.meta.get("channels"))


if __name__ == "__main__":
    main()
