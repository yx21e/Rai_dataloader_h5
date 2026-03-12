from __future__ import annotations

import glob
import json
import os
from typing import List, Optional

import geopandas as gpd
import pandas as pd

from dataloader.config import TimeConfig
from dataloader.normalize import rasterize_polygons_daily
from dataloader.schema import Grid


class NOAAAdapter:
    """Load NOAA NWM flood warnings GeoJSON and rasterize to grid."""

    DATA_DIR = "NOAA-NWMflood"

    def _list_files(self, root_dir: str) -> List[str]:
        base = os.path.join(root_dir, self.DATA_DIR)
        return sorted(glob.glob(os.path.join(base, "fl_flood_*.geojson")))

    def _select_files(self, files: List[str], time_cfg: TimeConfig) -> List[str]:
        if not time_cfg.start_date and not time_cfg.end_date:
            return files
        start = time_cfg.start_date or "1900-01-01"
        end = time_cfg.end_date or "2999-12-31"
        sy, sm = int(start[:4]), int(start[5:7])
        ey, em = int(end[:4]), int(end[5:7])

        selected = []
        for f in files:
            base = os.path.basename(f)
            parts = base.replace("fl_flood_", "").replace(".geojson", "").split("_")
            if len(parts) != 2:
                continue
            y, mo = int(parts[0]), int(parts[1])
            if (y, mo) < (sy, sm) or (y, mo) > (ey, em):
                continue
            selected.append(f)
        return selected

    def load(
        self,
        root_dir: str,
        time_cfg: TimeConfig,
        bbox: tuple[float, float, float, float],
        resolution: float,
    ) -> Optional[Grid]:
        files = self._select_files(self._list_files(root_dir), time_cfg)
        if not files:
            return None

        frames = []
        for path in files:
            gdf = gpd.read_file(path)
            if gdf.empty:
                continue
            if "issue" in gdf.columns:
                gdf["event_date"] = gdf["issue"].apply(lambda v: str(v)[:10])
            elif "polygon_begin" in gdf.columns:
                gdf["event_date"] = gdf["polygon_begin"].apply(lambda v: str(v)[:10])
            else:
                gdf["event_date"] = None
            frames.append(gdf[["geometry", "event_date"]])

        if not frames:
            return None

        df = pd.concat(frames, ignore_index=True)
        df = df[df["event_date"].notna()]
        return rasterize_polygons_daily(df, bbox=bbox, resolution=resolution, date_column="event_date")
