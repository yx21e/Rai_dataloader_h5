#!/usr/bin/env python3

from dataloader.adapters import firms as firms_mod
from dataloader import GeoLoadInput, load_data, load_sample_h5, save_sample_h5

import json
import warnings
from pathlib import Path

import numpy as np


OUT_DIR = Path("/home/yangshuang/output/publish_readiness")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _ratio(mask):
    arr = np.asarray(mask) if mask is not None else None
    if arr is None or arr.size == 0:
        return None
    return float(arr.mean())


def _roundtrip_h5(sample, stem: str):
    path = OUT_DIR / f"{stem}.h5"
    save_sample_h5(sample, str(path))
    restored = load_sample_h5(str(path))
    return {
        "path": str(path),
        "x_nonzero": int(np.count_nonzero(restored.x)),
        "y_nonzero": int(np.count_nonzero(restored.y)),
    }


def _run_case(name: str, request: GeoLoadInput):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        sample = load_data(request)

    warning_messages = [str(w.message).strip() for w in caught]
    result = {
        "name": name,
        "status": "fail"
        if any(marker in msg for msg in warning_messages for marker in ("[Data Source]", "[Feature Output]"))
        else "pass",
        "x_shape": tuple(int(v) for v in sample.x.shape),
        "y_shape": tuple(int(v) for v in sample.y.shape),
        "channels": list(sample.meta.get("channels", [])),
        "x_nonzero": int(np.count_nonzero(sample.x)),
        "y_nonzero": int(np.count_nonzero(sample.y)),
        "x_synthetic_ratio": _ratio(sample.meta.get("x_synthetic_mask")),
        "y_synthetic_ratio": _ratio(sample.meta.get("y_synthetic_mask")),
        "warnings": warning_messages,
        "h5": _roundtrip_h5(sample, name),
    }
    return result


def main() -> None:
    firms_mod.FIRMSAdapter.CSV_PATTERNS = ["firmsFL14-25/*.csv"]
    firms_mod.FIRMSAdapter.JSON_PATTERNS = []
    bbox = (-87.8, 24.0, -79.8, 31.5)

    cases = [
        ("firms_observed", GeoLoadInput(["FIRMS"], ("2023-01-01", "2023-01-03"), bbox, 0.25, synthetic_time=False, target_hazards=["wildfire"], label_source="firms")),
        ("firms_synthetic", GeoLoadInput(["FIRMS"], ("2023-01-01", "2023-01-10"), bbox, 0.25, synthetic_time=True, target_hazards=["wildfire"], label_source="firms")),
        ("noaa_observed", GeoLoadInput(["NOAA"], ("2023-01-01", "2023-01-10"), bbox, 0.25, synthetic_time=False, target_hazards=["flood"], label_source="noaa")),
        ("noaa_synthetic", GeoLoadInput(["NOAA"], ("2023-01-01", "2023-01-10"), bbox, 0.25, synthetic_time=True, target_hazards=["flood"], label_source="noaa")),
        ("era5_observed", GeoLoadInput(["ERA5"], ("2025-09-01", "2025-09-03"), bbox, 0.25, synthetic_time=False)),
        ("era5_synthetic", GeoLoadInput(["ERA5"], ("2025-09-01", "2025-09-02"), bbox, 0.25, synthetic_time=True, temporal_cadence="12h")),
        ("landfire_observed", GeoLoadInput(["LANDFIRE"], ("2023-01-01", "2023-01-02"), bbox, 0.25, synthetic_time=False)),
        ("landfire_synthetic", GeoLoadInput(["LANDFIRE"], ("2023-01-01", "2023-01-03"), bbox, 0.25, synthetic_time=True)),
        ("mtbs_label_observed", GeoLoadInput(["ERA5", "MTBS"], ("2024-07-01", "2024-07-10"), bbox, 0.25, synthetic_time=False, target_hazards=["wildfire"], label_source="mtbs")),
        ("mtbs_label_synthetic", GeoLoadInput(["ERA5", "MTBS"], ("2024-07-01", "2024-07-10"), bbox, 0.25, synthetic_time=True, target_hazards=["wildfire"], label_source="mtbs")),
        ("merra2_observed", GeoLoadInput(["MERRA2"], ("2023-01-01", "2023-01-03"), bbox, 0.25, synthetic_time=False)),
        ("merra2_synthetic", GeoLoadInput(["MERRA2"], ("2023-01-01", "2023-01-05"), bbox, 0.25, synthetic_time=True)),
    ]

    results = [_run_case(name, req) for name, req in cases]
    report_path = OUT_DIR / "publish_readiness_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("publish readiness summary")
    for item in results:
        if item["status"] != "pass":
            detail = item["warnings"][0].splitlines()[1] if item["warnings"] else "failed"
            print(f"- {item['name']}: FAIL | {detail}")
            continue
        print(
            f"- {item['name']}: PASS | x_shape={item['x_shape']} y_shape={item['y_shape']} "
            f"x_nonzero={item['x_nonzero']} y_nonzero={item['y_nonzero']} "
            f"h5_x_nonzero={item['h5']['x_nonzero']} h5_y_nonzero={item['h5']['y_nonzero']}"
        )
    print("report saved:", report_path)


if __name__ == "__main__":
    main()
