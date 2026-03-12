from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np

from dataloader.schema import Grid
from dataloader.normalize import make_grid


class LandfireAdapter:
    """Load LANDFIRE GeoTIFF and reproject to target bbox/resolution."""

    DATA_DIR = "USDALandfire/tifs"

    def _pick_latest(self, root_dir: str) -> Optional[str]:
        base = os.path.join(root_dir, self.DATA_DIR)
        files = sorted(glob.glob(os.path.join(base, "**", "Tif", "*.tif"), recursive=True))
        return files[-1] if files else None

    def load(
        self,
        root_dir: str,
        bbox: tuple[float, float, float, float],
        resolution: float,
    ) -> Optional[Grid]:
        try:
            import rasterio
            from rasterio.transform import from_origin
            from rasterio.warp import reproject, Resampling
        except ImportError as exc:
            raise ImportError("rasterio is required to load LANDFIRE GeoTIFFs.") from exc

        path = self._pick_latest(root_dir)
        if path is None:
            return None

        grid = make_grid(bbox, resolution)
        h, w = len(grid["lat"]), len(grid["lon"])
        dst = np.zeros((h, w), dtype="float32")
        dst_transform = from_origin(bbox[0], bbox[3], resolution, resolution)

        with rasterio.open(path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.nearest,
            )

        return Grid(
            data=dst[None, None, :, :].astype("float32"),
            coords={"time": np.array([]), "lat": grid["lat"], "lon": grid["lon"]},
            attrs={"channels": ["landfire_fuel"]},
        )
