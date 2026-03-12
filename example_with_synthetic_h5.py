#!/usr/bin/env python3

from dataloader import GeoLoadInput, load_data, load_sample_h5, save_sample_h5, to_torch_batch
from dataloader.adapters import firms as firms_mod


def main() -> None:
    firms_mod.FIRMSAdapter.CSV_PATTERNS = ["firmsFL14-25/*.csv"]
    firms_mod.FIRMSAdapter.JSON_PATTERNS = []

    request = GeoLoadInput(
        root_dir="/home/yangshuang",
        data_sources=["FIRMS"],
        temporal_window=("2023-01-01", "2023-01-02"),
        area_of_interest_bbox=(-87.8, 24.0, -79.8, 31.5),
        spatial_resolution_deg=0.25,
        synthetic_time=True,
        temporal_cadence="D",
        target_hazards=["wildfire"],
    )
    sample = load_data(request)

    print("load status: completed. reporting normalized output dimensions.")
    print("x shape:", tuple(sample.x.shape))
    print("y shape:", tuple(sample.y.shape))
    print("channels:", list(sample.meta.get("channels", [])))

    out_path = "/home/yangshuang/output/sample.h5"
    save_sample_h5(sample, out_path)
    print(f"artifact status: HDF5 dataset persisted at {out_path}")

    restored = load_sample_h5(out_path)
    x_t, y_t, meta = to_torch_batch(restored)
    print("torch tensors:", tuple(x_t.shape), tuple(y_t.shape), meta.get("channels", []))


if __name__ == "__main__":
    main()
