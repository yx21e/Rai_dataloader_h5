from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from dataloader.config import TimeConfig
from dataloader.normalize import parse_date


class MTBSFODAdapter:
    """Load MTBS Fire Occurrence Dataset (FOD) points."""

    SHP_PATH = "MTBS/Fire_Occurrence_Dataset/mtbs_FODpoints_DD.shp"

    def __init__(self, label_map: Optional[Dict[str, int]] = None) -> None:
        self.label_map = label_map or {}

    def _normalize_type(self, value: str | None) -> int:
        if value is None:
            return self.label_map.get("unk", 0)
        key = str(value).strip().lower()
        return self.label_map.get(key, self.label_map.get("unk", 0))

    def _apply_time_filter(self, df: pd.DataFrame, time_cfg: TimeConfig) -> pd.DataFrame:
        if df.empty or "Ig_Date" not in df.columns:
            return df
        df = df.copy()
        df["date"] = df["Ig_Date"].apply(lambda v: str(v) if v is not None else "")
        df["dt"] = pd.to_datetime(df["date"], errors="coerce")
        start = parse_date(time_cfg.start_date) if time_cfg.start_date else None
        end = parse_date(time_cfg.end_date) if time_cfg.end_date else None
        if start:
            df = df[df["dt"] >= start]
        if end:
            df = df[df["dt"] <= end]
        df["date"] = df["dt"].dt.strftime("%Y-%m-%d")
        df = df.drop(columns=["dt"])
        return df

    def load_points(self, root_dir: str, time_cfg: TimeConfig) -> pd.DataFrame:
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError("geopandas is required to load MTBS FOD shapefiles.") from exc

        shp_path = f"{root_dir}/{self.SHP_PATH}"
        gdf = gpd.read_file(shp_path)

        if gdf.empty:
            return pd.DataFrame(columns=["latitude", "longitude", "date", "label"])

        df = gdf.copy()
        df["longitude"] = df.geometry.x
        df["latitude"] = df.geometry.y
        df = self._apply_time_filter(df, time_cfg)

        if "Incid_Type" not in df.columns:
            df["Incid_Type"] = None

        df["label"] = df["Incid_Type"].apply(self._normalize_type)
        return df[["latitude", "longitude", "date", "label"]].dropna(subset=["latitude", "longitude"])
