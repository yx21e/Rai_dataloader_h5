#!/usr/bin/env python3

from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader.adapters import firms as firms_mod
from dataloader.registry import get_adapter
from dataloader_v2.config import TimeConfig
from dataloader_v4 import save_points_png


def main() -> None:
    bbox = (-87.8, 24.0, -79.8, 31.5)
    firms_mod.FIRMSAdapter.CSV_PATTERNS = ["firmsFL14-25/*.csv"]
    firms_mod.FIRMSAdapter.JSON_PATTERNS = []

    df = get_adapter("firms")().load_points(
        "/home/yangshuang",
        TimeConfig(start_date="2023-01-01", end_date="2023-01-05", frequency="D"),
    )
    df = df[
        (df["longitude"] >= bbox[0])
        & (df["longitude"] <= bbox[2])
        & (df["latitude"] >= bbox[1])
        & (df["latitude"] <= bbox[3])
    ].copy()

    out = save_points_png(
        df,
        output_path="/home/yangshuang/output/firms_points_frp.png",
        bbox=bbox,
        color_column="frp",
        title="FIRMS detections colored by FRP",
    )
    print("visualization saved:", out)


if __name__ == "__main__":
    main()
