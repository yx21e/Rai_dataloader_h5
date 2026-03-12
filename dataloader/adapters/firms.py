from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List

import pandas as pd

from dataloader.catalog import build_catalog
from dataloader.config import TimeConfig
from dataloader.normalize import parse_date


class FIRMSAdapter:
    """Load FIRMS fire detections (CSV/JSON)."""

    CSV_PATTERNS = ["firmsFL14-25/*.csv", "firms14-25/*.csv"]
    JSON_PATTERNS = ["firms14-25/*.json", "firmsFL14-25/*.json"]
    DATA_YEAR_MIN = 2014
    DATA_YEAR_MAX = 2025

    def _is_outside_supported_years(self, time_cfg: TimeConfig) -> bool:
        if not time_cfg.start_date and not time_cfg.end_date:
            return False
        start_year = int(time_cfg.start_date[:4]) if time_cfg.start_date else self.DATA_YEAR_MIN
        end_year = int(time_cfg.end_date[:4]) if time_cfg.end_date else self.DATA_YEAR_MAX
        return end_year < self.DATA_YEAR_MIN or start_year > self.DATA_YEAR_MAX

    def list_assets(self, root_dir: str) -> List[str]:
        assets = build_catalog(root_dir, "firms", self.CSV_PATTERNS + self.JSON_PATTERNS)
        return [a.path for a in assets]

    def _load_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        return df

    def _load_json(self, path: str) -> pd.DataFrame:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict) and "features" in payload:
            rows = [f.get("properties", {}) for f in payload["features"]]
            return pd.DataFrame(rows)
        raise ValueError(f"Unsupported JSON format: {path}")

    def _apply_time_filter(self, df: pd.DataFrame, time_cfg: TimeConfig) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        if "acq_date" in df.columns:
            df["date"] = df["acq_date"].apply(lambda v: str(v) if v is not None else "")
        elif "date" in df.columns:
            df["date"] = df["date"].apply(lambda v: str(v) if v is not None else "")
        else:
            return df
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
        # Fast-path for clearly out-of-range requests (avoids parsing huge FIRMS JSON files).
        if self._is_outside_supported_years(time_cfg):
            return pd.DataFrame(columns=["latitude", "longitude", "date", "frp"])

        frames = []
        for path in self.list_assets(root_dir):
            if path.endswith(".csv"):
                frames.append(self._load_csv(path))
            elif path.endswith(".json"):
                frames.append(self._load_json(path))
        if not frames:
            return pd.DataFrame(columns=["latitude", "longitude", "date", "frp"])
        df = pd.concat(frames, ignore_index=True)
        if "acq_date" not in df.columns and "date" not in df.columns:
            raise KeyError("FIRMS data must include acq_date or date.")
        if "date" not in df.columns:
            df["date"] = df["acq_date"]
        df = self._apply_time_filter(df, time_cfg)

        for col in ("latitude", "longitude", "frp", "type", "confidence"):
            if col not in df.columns:
                df[col] = None

        return df[["latitude", "longitude", "date", "frp", "type", "confidence"]].dropna(
            subset=["latitude", "longitude"]
        )
