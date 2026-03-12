from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dataloader.cache import load_grid, save_grid
from dataloader.config import PipelineConfig
from dataloader.normalize import rasterize_labels_daily, rasterize_points_daily
from dataloader.registry import get_adapter
from dataloader.schema import Grid, Sample


class UnifiedDataLoader:
    """Unified dataloader for multi-source wildfire data."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def _ensure_bbox(self, frames: List[pd.DataFrame]) -> tuple[float, float, float, float]:
        if self.config.grid.bbox is not None:
            return self.config.grid.bbox
        mins, maxs = [], []
        for df in frames:
            if df.empty:
                continue
            mins.append((df["longitude"].min(), df["latitude"].min()))
            maxs.append((df["longitude"].max(), df["latitude"].max()))
        if not mins:
            raise ValueError("bbox is required when no points are available to infer it.")
        minx = min(v[0] for v in mins)
        miny = min(v[1] for v in mins)
        maxx = max(v[0] for v in maxs)
        maxy = max(v[1] for v in maxs)
        return (float(minx), float(miny), float(maxx), float(maxy))

    def _bbox_from_grid(self, grid: Grid) -> tuple[float, float, float, float]:
        if self.config.grid.bbox is not None:
            return self.config.grid.bbox
        lon = grid.coords.get("lon")
        lat = grid.coords.get("lat")
        if lon is None or lat is None or len(lon) == 0 or len(lat) == 0:
            raise ValueError("Cannot infer bbox from cached grid without coordinates.")
        res = self.config.grid.resolution_deg
        minx = float(lon.min() - res / 2.0)
        maxx = float(lon.max() + res / 2.0)
        miny = float(lat.min() - res / 2.0)
        maxy = float(lat.max() + res / 2.0)
        return (minx, miny, maxx, maxy)

    def _load_firms(self, bbox: tuple[float, float, float, float]) -> Optional[Grid]:
        adapter = get_adapter("firms")()
        df = adapter.load_points(self.config.root_dir, self.config.time)
        if df.empty:
            return None
        return rasterize_points_daily(
            df,
            bbox=bbox,
            resolution=self.config.grid.resolution_deg,
            value_columns=["frp"],
        )

    def _load_mtbs_fod(self, bbox: tuple[float, float, float, float]) -> Optional[Grid]:
        adapter = get_adapter("mtbs_fod")(label_map=self.config.label.incid_type_map)
        df = adapter.load_points(self.config.root_dir, self.config.time)
        if df.empty:
            return None
        return rasterize_labels_daily(
            df,
            bbox=bbox,
            resolution=self.config.grid.resolution_deg,
            label_column="label",
            label_rule=self.config.label.label_rule,
        )

    def _align_feature_grids(self, grids: List[Grid]) -> Grid:
        if not grids:
            raise ValueError("No feature grids available to align.")
        target_dates = self._timeline_from_config()

        if not target_dates:
            raise ValueError("No overlapping time coordinates across sources.")

        target_times = np.array([np.datetime64(d) for d in target_dates])
        aligned = []
        for grid in grids:
            data = grid.data
            times = grid.coords.get("time")
            c = data.shape[1] if data.ndim == 4 else 1
            h = data.shape[-2]
            w = data.shape[-1]
            if times is None or len(times) == 0:
                # Static grid, tile across time.
                if data.ndim == 2:
                    data = data[None, None, :, :]
                elif data.ndim == 3:
                    data = data[None, :, :, :]
                data = np.repeat(data, len(target_times), axis=0)
            else:
                padded = np.zeros((len(target_times), c, h, w), dtype=data.dtype)
                mapping = {str(pd.to_datetime(t).date()): i for i, t in enumerate(times)}
                for i, d in enumerate(target_dates):
                    j = mapping.get(d)
                    if j is not None:
                        padded[i, ...] = data[j, ...]
                data = padded
            aligned.append(data)

        h, w = aligned[0].shape[-2], aligned[0].shape[-1]
        channels = []
        for grid in grids:
            channels.extend(grid.attrs.get("channels", []))

        stacked = np.concatenate(aligned, axis=1) if len(aligned) > 1 else aligned[0]
        return Grid(
            data=stacked,
            coords={"time": target_times, "lat": grids[0].coords.get("lat"), "lon": grids[0].coords.get("lon")},
            attrs={"channels": channels},
        )

    def _timeline_from_config(self) -> List[str]:
        if not self.config.time.start_date or not self.config.time.end_date:
            raise ValueError("date_range is required when selected feature sources have no time dimension.")
        start = pd.to_datetime(self.config.time.start_date).date()
        end = pd.to_datetime(self.config.time.end_date).date()
        if end < start:
            raise ValueError("end_date must be >= start_date.")
        days = []
        cur = start
        while cur <= end:
            days.append(cur.strftime("%Y-%m-%d"))
            cur += timedelta(days=1)
        return days

    def _align_label_grid(self, label_grid: Grid, feature_grid: Grid) -> Grid:
        """Align labels to feature timeline with zero-fill for missing dates."""
        x_times = feature_grid.coords.get("time")
        if x_times is None:
            raise ValueError("Feature grid missing time coordinate.")

        t, h, w = feature_grid.data.shape[0], feature_grid.data.shape[-2], feature_grid.data.shape[-1]
        out = np.zeros((t, h, w), dtype="int16")

        y_times = label_grid.coords.get("time")
        if y_times is None or len(y_times) == 0 or label_grid.data.size == 0:
            return Grid(
                data=out,
                coords={"time": x_times, "lat": feature_grid.coords.get("lat"), "lon": feature_grid.coords.get("lon")},
                attrs=label_grid.attrs,
            )

        x_map = {str(pd.to_datetime(ts).date()): i for i, ts in enumerate(x_times)}
        y_map = {str(pd.to_datetime(ts).date()): i for i, ts in enumerate(y_times)}

        common = set(x_map).intersection(y_map)
        for d in common:
            out[x_map[d], :, :] = label_grid.data[y_map[d], :, :]

        return Grid(
            data=out,
            coords={"time": x_times, "lat": feature_grid.coords.get("lat"), "lon": feature_grid.coords.get("lon")},
            attrs=label_grid.attrs,
        )

    def build(self) -> Sample:
        """Return unified sample with x [T, C, H, W] and y [T, H, W]."""
        cached = None
        if self.config.cache_dir:
            cached = load_grid(self.config.cache_dir, "features")

        firms_grid = None
        bbox = self.config.grid.bbox
        if cached is None:
            firms_df = get_adapter("firms")().load_points(self.config.root_dir, self.config.time)
            mtbs_df = get_adapter("mtbs_fod")(label_map=self.config.label.incid_type_map).load_points(
                self.config.root_dir, self.config.time
            )
            if bbox is None:
                bbox = self._ensure_bbox([firms_df, mtbs_df])
            feature_grids: List[Grid] = []
            if "firms" in self.config.sources:
                firms_grid = rasterize_points_daily(
                    firms_df,
                    bbox=bbox,
                    resolution=self.config.grid.resolution_deg,
                    value_columns=["frp"],
                )
                feature_grids.append(firms_grid)
            if "era5" in self.config.sources:
                era5_grid = get_adapter("era5")().load(
                    self.config.root_dir,
                    self.config.time,
                    bbox=bbox,
                    resolution=self.config.grid.resolution_deg,
                )
                if era5_grid is not None:
                    feature_grids.append(era5_grid)
            if "merra2" in self.config.sources:
                merra_grid = get_adapter("merra2")().load(
                    self.config.root_dir,
                    self.config.time,
                    bbox=bbox,
                    resolution=self.config.grid.resolution_deg,
                )
                if merra_grid is not None:
                    feature_grids.append(merra_grid)
            if "noaa" in self.config.sources:
                noaa_grid = get_adapter("noaa")().load(
                    self.config.root_dir,
                    self.config.time,
                    bbox=bbox,
                    resolution=self.config.grid.resolution_deg,
                )
                if noaa_grid is not None:
                    feature_grids.append(noaa_grid)
            if "landfire" in self.config.sources:
                landfire_grid = get_adapter("landfire")().load(
                    self.config.root_dir,
                    bbox=bbox,
                    resolution=self.config.grid.resolution_deg,
                )
                if landfire_grid is not None:
                    feature_grids.append(landfire_grid)

            if not feature_grids:
                raise ValueError("No feature sources returned data.")

            firms_grid = self._align_feature_grids(feature_grids)

            if self.config.cache_dir:
                save_grid(firms_grid, self.config.cache_dir, "features")
        else:
            firms_grid = cached
            bbox = self._bbox_from_grid(firms_grid)

        label_cached = None
        if self.config.cache_dir:
            label_cached = load_grid(self.config.cache_dir, "labels")
        if label_cached is None:
            if "mtbs_fod" in self.config.sources:
                mtbs_grid = self._load_mtbs_fod(bbox)
            else:
                mtbs_grid = None
            if mtbs_grid is None:
                if firms_grid is None:
                    raise ValueError("No sources produced data; check 'sources' in config.")
                mtbs_grid = Grid(
                    data=np.zeros((firms_grid.data.shape[0],) + firms_grid.data.shape[2:], dtype="int16"),
                    coords=firms_grid.coords,
                    attrs={"label_rule": self.config.label.label_rule},
                )
            if self.config.cache_dir:
                save_grid(mtbs_grid, self.config.cache_dir, "labels")
        else:
            mtbs_grid = label_cached

        if firms_grid is not None:
            mtbs_grid = self._align_label_grid(mtbs_grid, firms_grid)

        if firms_grid is None:
            h, w = mtbs_grid.data.shape[1:]
            x = np.zeros((mtbs_grid.data.shape[0], 0, h, w), dtype="float32")
        else:
            x = firms_grid.data
        y = mtbs_grid.data
        channels = firms_grid.attrs.get("channels", []) if firms_grid is not None else []
        meta = {
            "config": asdict(self.config),
            "channels": channels,
        }
        return Sample(x=x, y=y, meta=meta)
