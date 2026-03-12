#python /home/yangshuang/dataloader/example_simple_api.py


from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataloader import load_data


def main() -> None:
    sample = load_data(
        data=["FIRMS", "ERA5", "NOAA","MTBS"],
        date_range=("2023-01-01", "2023-01-31"),
        bbox=(-87.8, 24.0, -79.8, 31.5),
        resolution=0.25,
        root_dir="/home/yangshuang",
        # Keep cache disabled in this demo so output always reflects current logic.
        cache_dir=None,
    )

    print("x shape:", sample.x.shape)
    print("y shape:", sample.y.shape)
    print("channels:", sample.meta.get("channels"))


if __name__ == "__main__":
    main()
