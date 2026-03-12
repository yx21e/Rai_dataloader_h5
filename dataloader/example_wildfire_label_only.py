#python /home/yangshuang/dataloader/example_wildfire_label_only.py


from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader import load_data
from dataloader.adapters import firms as firms_mod


def main() -> None:
    wildfire_only_map = {
        "unk": 0,
        "unknown": 0,
        "": 0,
        "wf": 1,
        "wildfire": 1,
        "rx": 0,
        "prescribed": 0,
        "prescribed fire": 0,
        "wildland fire use": 0,
        "out of area response": 0,
        "complex": 0,
    }

    # Keep runtime bounded for local demo runs.
    firms_mod.FIRMSAdapter.CSV_PATTERNS = ["firmsFL14-25/*.csv"]
    firms_mod.FIRMSAdapter.JSON_PATTERNS = []

    sample = load_data(
        data=["FIRMS", "MTBS"],
        # Wider range to ensure MTBS wildfire labels are present.
        date_range=("2022-01-01", "2023-12-31"),
        bbox=(-87.8, 24.0, -79.8, 31.5),
        resolution=0.25,
        root_dir="/home/yangshuang",
        label_map=wildfire_only_map,
    )

    print("x shape:", sample.x.shape)
    print("y shape:", sample.y.shape)
    print("wildfire pixels:", int((sample.y == 1).sum()))
    print("non-wildfire pixels:", int((sample.y == 0).sum()))


if __name__ == "__main__":
    main()
