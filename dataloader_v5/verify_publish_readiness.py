from __future__ import annotations

import json
import warnings
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path("/home/yangshuang")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataloader.adapters import firms as firms_mod
from dataloader_v3 import GeoLoadInput, load_data, load_sample_h5, save_sample_h5


ROOT_DIR = str(ROOT)
OUT_DIR = Path("/home/yangshuang/output/publish_readiness")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _ratio(mask: Any) -> float | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.size == 0:
        return None
    return float(arr.mean())


def _roundtrip_h5(sample, stem: str) -> dict[str, Any]:
    path = OUT_DIR / f"{stem}.h5"
    save_sample_h5(sample, str(path))
    restored = load_sample_h5(str(path))
    return {
        "path": str(path),
        "x_nonzero": int(np.count_nonzero(restored.x)),
        "y_nonzero": int(np.count_nonzero(restored.y)),
        "x_shape": tuple(int(v) for v in restored.x.shape),
        "y_shape": tuple(int(v) for v in restored.y.shape),
    }


def _run_case(name: str, request: GeoLoadInput) -> dict[str, Any]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        sample = load_data(request)

    meta = sample.meta
    h5 = _roundtrip_h5(sample, name)

    warning_messages = [str(w.message).strip() for w in caught]
    blocking_warning_markers = ("[Data Source]", "[Feature Output]")

    return {
        "name": name,
        "status": "fail"
        if any(marker in msg for msg in warning_messages for marker in blocking_warning_markers)
        else "pass",
        "request": {
            "data_sources": list(request.data_sources),
            "temporal_window": tuple(request.temporal_window),
            "bbox": tuple(request.area_of_interest_bbox),
            "spatial_resolution_deg": request.spatial_resolution_deg,
            "synthetic_time": request.synthetic_time,
            "temporal_cadence": request.temporal_cadence,
            "target_hazards": request.target_hazards,
            "label_source": request.label_source,
        },
        "x_shape": tuple(int(v) for v in sample.x.shape),
        "y_shape": tuple(int(v) for v in sample.y.shape),
        "channels": list(meta.get("channels", [])),
        "x_nonzero": int(np.count_nonzero(sample.x)),
        "y_nonzero": int(np.count_nonzero(sample.y)),
        "x_synthetic_ratio": _ratio(meta.get("x_synthetic_mask")),
        "y_synthetic_ratio": _ratio(meta.get("y_synthetic_mask")),
        "warnings": warning_messages,
        "h5": h5,
    }


def main() -> None:
    firms_mod.FIRMSAdapter.CSV_PATTERNS = ["firmsFL14-25/*.csv"]
    firms_mod.FIRMSAdapter.JSON_PATTERNS = []

    bbox = (-87.8, 24.0, -79.8, 31.5)
    cases = [
        (
            "firms_observed",
            GeoLoadInput(
                data_sources=["FIRMS"],
                temporal_window=("2023-01-01", "2023-01-03"),
                area_of_interest_bbox=bbox,
                spatial_resolution_deg=0.25,
                synthetic_time=False,
                temporal_cadence="D",
                target_hazards=["wildfire"],
                label_source="firms",
            ),
        ),
        (
            "firms_synthetic",
            GeoLoadInput(
                data_sources=["FIRMS"],
                temporal_window=("2023-01-01", "2023-01-10"),
                area_of_interest_bbox=bbox,
                spatial_resolution_deg=0.25,
                synthetic_time=True,
                temporal_cadence="D",
                target_hazards=["wildfire"],
                label_source="firms",
            ),
        ),
        (
            "noaa_observed",
            GeoLoadInput(
                data_sources=["NOAA"],
                temporal_window=("2023-01-01", "2023-01-10"),
                area_of_interest_bbox=bbox,
                spatial_resolution_deg=0.25,
                synthetic_time=False,
                temporal_cadence="D",
                target_hazards=["flood"],
                label_source="noaa",
            ),
        ),
        (
            "noaa_synthetic",
            GeoLoadInput(
                data_sources=["NOAA"],
                temporal_window=("2023-01-01", "2023-01-10"),
                area_of_interest_bbox=bbox,
                spatial_resolution_deg=0.25,
                synthetic_time=True,
                temporal_cadence="D",
                target_hazards=["flood"],
                label_source="noaa",
            ),
        ),
        (
            "era5_observed",
            GeoLoadInput(
                data_sources=["ERA5"],
                temporal_window=("2025-09-01", "2025-09-03"),
                area_of_interest_bbox=bbox,
                spatial_resolution_deg=0.25,
                synthetic_time=False,
                temporal_cadence="D",
            ),
        ),
        (
            "era5_synthetic",
            GeoLoadInput(
                data_sources=["ERA5"],
                temporal_window=("2025-09-01", "2025-09-02"),
                area_of_interest_bbox=bbox,
                spatial_resolution_deg=0.25,
                synthetic_time=True,
                temporal_cadence="12H",
            ),
        ),
        (
            "landfire_observed",
            GeoLoadInput(
                data_sources=["LANDFIRE"],
                temporal_window=("2023-01-01", "2023-01-02"),
                area_of_interest_bbox=bbox,
                spatial_resolution_deg=0.25,
                synthetic_time=False,
                temporal_cadence="D",
            ),
        ),
        (
            "landfire_synthetic",
            GeoLoadInput(
                data_sources=["LANDFIRE"],
                temporal_window=("2023-01-01", "2023-01-03"),
                area_of_interest_bbox=bbox,
                spatial_resolution_deg=0.25,
                synthetic_time=True,
                temporal_cadence="D",
            ),
        ),
        (
            "mtbs_label_observed",
            GeoLoadInput(
                data_sources=["ERA5", "MTBS"],
                temporal_window=("2024-07-01", "2024-07-10"),
                area_of_interest_bbox=bbox,
                spatial_resolution_deg=0.25,
                synthetic_time=False,
                temporal_cadence="D",
                target_hazards=["wildfire"],
                label_source="mtbs",
            ),
        ),
        (
            "mtbs_label_synthetic",
            GeoLoadInput(
                data_sources=["ERA5", "MTBS"],
                temporal_window=("2024-07-01", "2024-07-10"),
                area_of_interest_bbox=bbox,
                spatial_resolution_deg=0.25,
                synthetic_time=True,
                temporal_cadence="D",
                target_hazards=["wildfire"],
                label_source="mtbs",
            ),
        ),
        (
            "merra2_observed",
            GeoLoadInput(
                data_sources=["MERRA2"],
                temporal_window=("2023-01-01", "2023-01-03"),
                area_of_interest_bbox=bbox,
                spatial_resolution_deg=0.25,
                synthetic_time=False,
                temporal_cadence="D",
            ),
        ),
        (
            "merra2_synthetic",
            GeoLoadInput(
                data_sources=["MERRA2"],
                temporal_window=("2023-01-01", "2023-01-05"),
                area_of_interest_bbox=bbox,
                spatial_resolution_deg=0.25,
                synthetic_time=True,
                temporal_cadence="D",
            ),
        ),
    ]

    results = []
    for name, request in cases:
        print(f"[RUN] {name}")
        try:
            result = _run_case(name, request)
        except Exception as exc:
            result = {
                "name": name,
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
            }
        results.append(result)

    report_path = OUT_DIR / "publish_readiness_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\npublish readiness summary")
    for item in results:
        if item["status"] != "pass":
            detail = item.get("error")
            if detail is None and item.get("warnings"):
                detail = item["warnings"][0].splitlines()[1] if "\n" in item["warnings"][0] else item["warnings"][0]
            print(f"- {item['name']}: FAIL | {detail}")
            continue
        print(
            f"- {item['name']}: PASS | x_shape={item['x_shape']} y_shape={item['y_shape']} "
            f"x_nonzero={item['x_nonzero']} y_nonzero={item['y_nonzero']} h5_x_nonzero={item['h5']['x_nonzero']} "
            f"h5_y_nonzero={item['h5']['y_nonzero']}"
        )

    print(f"\nreport saved: {report_path}")


if __name__ == "__main__":
    main()
