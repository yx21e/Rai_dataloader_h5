from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
import warnings
import glob
import os
import re

import numpy as np
import pandas as pd
import xarray as xr

from dataloader.normalize import make_grid, rasterize_labels_daily, rasterize_points_daily
from dataloader.registry import get_adapter
from dataloader.schema import Grid, Sample
from dataloader.config import PipelineConfig


SOURCE_TYPE = {
    "firms": "event",
    "noaa": "event",
    "era5": "continuous",
    "merra2": "continuous",
    "landfire": "static",
    "mtbs_fod": "label",
}

NOAA_HAZARD_MAP = {
    "FL": "flood",
    "FF": "flood",
    "FA": "flood",
    "SV": "thunderstorm",
    "TO": "tornado",
    "MA": "marine",
}

MTBS_INCID_ALIAS = {
    "wf": "wildfire",
    "wildfire": "wildfire",
    "rx": "prescribed_fire",
    "prescribed": "prescribed_fire",
    "prescribed fire": "prescribed_fire",
    "wildland fire use": "wildland_fire_use",
    "unknown": "unknown_fire",
    "unk": "unknown_fire",
    "out of area response": "out_of_area_response",
    "complex": "complex_fire",
}


class UnifiedDataLoader:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._availability_cache: Dict[str, bool] = {}

    def _selected_hazards(self) -> set[str]:
        if self.config.label.label_hazards:
            return {h.lower() for h in self.config.label.label_hazards}
        return {k.lower() for k in self.config.label.label_mapping.keys()}

    def _label_id(self, hazard: str) -> int:
        key = str(hazard).strip().lower()
        if key not in self._selected_hazards():
            return self.config.label.label_default_value
        return int(self.config.label.label_mapping.get(key, self.config.label.label_default_value))

    def _timeline_from_config(self) -> list[str]:
        if not self.config.time.start_date or not self.config.time.end_date:
            raise ValueError("date_range is required.")
        idx = pd.date_range(
            start=self.config.time.start_date,
            end=self.config.time.end_date,
            freq=self.config.synthetic.target_freq,
        )
        return [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in idx]

    def _ensure_bbox(self, frames: List[pd.DataFrame]) -> Tuple[float, float, float, float]:
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
        return (
            float(min(v[0] for v in mins)),
            float(min(v[1] for v in mins)),
            float(max(v[0] for v in maxs)),
            float(max(v[1] for v in maxs)),
        )

    def _warn_missing_source(self, source: str) -> None:
        msg = (
            "\n#### Warning ###\n"
            f"[Data Source] '{source}' has no available records for the current query.\n"
            "[Action] Revise time range, AOI (bbox), or selected sources; alternatively enable synthetic_time.\n"
            "[Note] Pipeline execution continues by design; this warning indicates data unavailability, not a runtime failure.\n"
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

    def _date_bounds(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        if not self.config.time.start_date or not self.config.time.end_date:
            raise ValueError("date_range is required.")
        start = pd.to_datetime(self.config.time.start_date)
        end = pd.to_datetime(self.config.time.end_date)
        if end < start:
            raise ValueError("end_date must be >= start_date.")
        return start, end

    def _months_in_range(self) -> set[Tuple[int, int]]:
        start, end = self._date_bounds()
        idx = pd.period_range(start=start, end=end, freq="M")
        return {(p.year, p.month) for p in idx}

    def _source_maybe_available(self, source: str) -> bool:
        if source in self._availability_cache:
            return self._availability_cache[source]

        root = self.config.root_dir
        months_needed = self._months_in_range()

        if source == "firms":
            start, end = self._date_bounds()
            ok = not (end.year < 2014 or start.year > 2025)
        elif source == "era5":
            files: list[str] = []
            for d in ("era5/era5_data", "era5/era510", "era5/era59", "era5/era5new"):
                files.extend(glob.glob(os.path.join(root, d, "era5_*.nc")))
            have = set()
            for f in files:
                m = re.search(r"era5_(\d{4})_(\d{2})", os.path.basename(f))
                if m:
                    have.add((int(m.group(1)), int(m.group(2))))
            ok = len(have.intersection(months_needed)) > 0
        elif source == "noaa":
            files = glob.glob(os.path.join(root, "NOAA-NWMflood", "fl_flood_*.geojson"))
            have = set()
            for f in files:
                m = re.search(r"fl_flood_(\d{4})_(\d{2})\.geojson$", os.path.basename(f))
                if m:
                    have.add((int(m.group(1)), int(m.group(2))))
            ok = len(have.intersection(months_needed)) > 0
        elif source == "merra2":
            # Lightweight existence check (many environments have no local MERRA2 files).
            ok = len(glob.glob(os.path.join(root, "merra2", "**", "*.nc"), recursive=True)) > 0
        elif source == "landfire":
            ok = len(glob.glob(os.path.join(root, "USDALandfire", "tifs", "**", "Tif", "*.tif"), recursive=True)) > 0
        elif source == "mtbs_fod":
            ok = os.path.exists(os.path.join(root, "MTBS", "Fire_Occurrence_Dataset", "mtbs_FODpoints_DD.shp"))
        else:
            ok = True

        self._availability_cache[source] = ok
        return ok

    def _warn_no_features(self) -> None:
        msg = (
            "\n#### Warning ###\n"
            "[Feature Output] No usable feature tensor could be constructed from the selected sources.\n"
            "[Action] Revise query parameters (time range, AOI, source set), or enable synthetic_time for temporal gap filling.\n"
            "[Note] API contract is preserved: x is returned as (T, 0, H, W) and y as (T, H, W).\n"
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

    def _warn_missing_labels(self, source: Optional[str]) -> None:
        msg = (
            "\n#### Warning ###\n"
            f"[Label Source] '{source}' produced no label records for the current query.\n"
            "[Action] Revise label_source, target hazards, time range, or AOI; otherwise choose an alternate label source.\n"
            "[Note] y is returned as the default-value grid with an accompanying synthetic label mask.\n"
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

    def _resample_grid(
        self, grid: Grid, target_index: pd.DatetimeIndex, method: str, source_kind: str
    ) -> Tuple[Grid, np.ndarray]:
        data = grid.data
        if data.ndim == 2:
            data = data[None, None, :, :]
        elif data.ndim == 3:
            data = data[None, :, :, :]

        times = grid.coords.get("time")
        if times is None or len(times) == 0:
            # Static source: tile through time axis.
            tiled = np.repeat(data, len(target_index), axis=0)
            # No explicit timestamps in source; expanded slices are synthetic in time.
            synth_t = np.ones(len(target_index), dtype=bool)
            return (
                Grid(
                    data=tiled.astype("float32"),
                    coords={
                        "time": target_index.values,
                        "lat": grid.coords.get("lat"),
                        "lon": grid.coords.get("lon"),
                    },
                    attrs=grid.attrs,
                ),
                synth_t,
            )

        da = xr.DataArray(
            data,
            dims=("time", "channel", "lat", "lon"),
            coords={
                "time": pd.to_datetime(times),
                "channel": np.arange(data.shape[1]),
                "lat": grid.coords.get("lat"),
                "lon": grid.coords.get("lon"),
            },
        ).sortby("time")

        src_index = pd.to_datetime(times)
        exact_obs = target_index.isin(src_index)

        if method == "nearest":
            out = da.reindex(time=target_index, method="nearest")
        elif method == "ffill":
            out = da.reindex(time=target_index, method="ffill").fillna(0)
        elif method == "linear":
            out = da.reindex(time=target_index).interpolate_na(dim="time", method="linear")
            if source_kind in ("event", "label"):
                out = out.fillna(0)
            else:
                out = out.ffill("time").bfill("time")
        else:
            raise ValueError(f"Unsupported synthetic method: {method}")

        return (
            Grid(
                data=out.values.astype("float32"),
                coords={"time": target_index.values, "lat": grid.coords.get("lat"), "lon": grid.coords.get("lon")},
                attrs=grid.attrs,
            ),
            ~exact_obs,
        )

    def _align_by_intersection(self, items: List[Tuple[str, Grid]]) -> Tuple[Grid, np.ndarray]:
        dynamic_dates: List[set[str]] = []
        for _, grid in items:
            times = grid.coords.get("time")
            if times is not None and len(times) > 0:
                dynamic_dates.append({str(pd.to_datetime(t)) for t in times})

        if dynamic_dates:
            target_dates = sorted(set.intersection(*dynamic_dates))
        else:
            target_dates = self._timeline_from_config()
        if not target_dates:
            raise ValueError("No overlapping dates across selected sources.")
        target_index = pd.to_datetime(target_dates)

        aligned_parts = []
        aligned_masks = []
        channels: list[str] = []
        for source, grid in items:
            kind = SOURCE_TYPE.get(source, "event")
            method = "ffill" if kind == "event" else "nearest"
            aligned, synth_t = self._resample_grid(grid, target_index=target_index, method=method, source_kind=kind)
            aligned_parts.append(aligned.data)
            c = aligned.data.shape[1]
            aligned_masks.append(np.repeat(synth_t[:, None], c, axis=1))
            channels.extend(aligned.attrs.get("channels", []))

        x = np.concatenate(aligned_parts, axis=1) if len(aligned_parts) > 1 else aligned_parts[0]
        x_mask = np.concatenate(aligned_masks, axis=1) if len(aligned_masks) > 1 else aligned_masks[0]
        ref = items[0][1]
        return (
            Grid(
                data=x,
                coords={"time": target_index.values, "lat": ref.coords.get("lat"), "lon": ref.coords.get("lon")},
                attrs={"channels": channels},
            ),
            x_mask,
        )

    def _align_synthetic(self, items: List[Tuple[str, Grid]]) -> Tuple[Grid, np.ndarray]:
        target_index = pd.to_datetime(self._timeline_from_config())
        aligned_parts = []
        aligned_masks = []
        channels: list[str] = []

        for source, grid in items:
            kind = SOURCE_TYPE.get(source, "event")
            if kind == "continuous":
                method = self.config.synthetic.continuous_method
            elif kind == "event":
                method = self.config.synthetic.event_method
            else:
                method = "ffill"

            aligned, synth_t = self._resample_grid(grid, target_index=target_index, method=method, source_kind=kind)
            aligned_parts.append(aligned.data)
            c = aligned.data.shape[1]
            aligned_masks.append(np.repeat(synth_t[:, None], c, axis=1))
            channels.extend(aligned.attrs.get("channels", []))

        x = np.concatenate(aligned_parts, axis=1) if len(aligned_parts) > 1 else aligned_parts[0]
        x_mask = np.concatenate(aligned_masks, axis=1) if len(aligned_masks) > 1 else aligned_masks[0]
        ref = items[0][1]
        return (
            Grid(
                data=x,
                coords={"time": target_index.values, "lat": ref.coords.get("lat"), "lon": ref.coords.get("lon")},
                attrs={"channels": channels},
            ),
            x_mask,
        )

    def _load_mtbs_labels(self, bbox: Tuple[float, float, float, float]) -> Optional[Grid]:
        # Build MTBS alias->id map based on client hazard selection/mapping.
        mtbs_map: Dict[str, int] = {}
        for alias, hazard in MTBS_INCID_ALIAS.items():
            mtbs_map[alias] = self._label_id(hazard)
        adapter = get_adapter("mtbs_fod")(label_map=mtbs_map)
        df = adapter.load_points(self.config.root_dir, self.config.time)
        if df.empty:
            return None
        return rasterize_labels_daily(
            df,
            bbox=bbox,
            resolution=self.config.grid.resolution_deg,
            label_column="label",
            label_rule="max",
        )

    def _load_firms_labels(self, bbox: Tuple[float, float, float, float]) -> Optional[Grid]:
        wildfire_id = self._label_id("wildfire")
        if wildfire_id == self.config.label.label_default_value:
            return None
        df = get_adapter("firms")().load_points(self.config.root_dir, self.config.time)
        if df.empty:
            return None
        # Optional quality filters for FIRMS-as-labels.
        if self.config.label.firms_type_allowlist and "type" in df.columns:
            allow = {str(v).strip() for v in self.config.label.firms_type_allowlist}
            df = df[df["type"].astype(str).str.strip().isin(allow)]
        if self.config.label.firms_min_confidence is not None and "confidence" in df.columns:
            conf = pd.to_numeric(df["confidence"], errors="coerce")
            df = df[conf >= float(self.config.label.firms_min_confidence)]
        if df.empty:
            return None
        points_grid = rasterize_points_daily(
            df,
            bbox=bbox,
            resolution=self.config.grid.resolution_deg,
            value_columns=[],
        )
        y = (points_grid.data[:, 0, :, :] > 0).astype("int16") * int(wildfire_id)
        return Grid(data=y, coords=points_grid.coords, attrs={"label_source": "firms"})

    def _list_noaa_files_in_range(self) -> List[str]:
        files = glob.glob(os.path.join(self.config.root_dir, "NOAA-NWMflood", "fl_flood_*.geojson"))
        months_needed = self._months_in_range()
        selected = []
        for f in files:
            m = re.search(r"fl_flood_(\d{4})_(\d{2})\.geojson$", os.path.basename(f))
            if not m:
                continue
            ym = (int(m.group(1)), int(m.group(2)))
            if ym in months_needed:
                selected.append(f)
        return sorted(selected)

    def _load_noaa_labels(self, bbox: Tuple[float, float, float, float]) -> Optional[Grid]:
        try:
            import geopandas as gpd
            from shapely.geometry import Point
            from shapely.prepared import prep
        except Exception:
            return None

        files = self._list_noaa_files_in_range()
        if not files:
            return None

        grid = make_grid(bbox, self.config.grid.resolution_deg)
        h, w = len(grid["lat"]), len(grid["lon"])
        points = [Point(lon, lat) for lat in grid["lat"] for lon in grid["lon"]]

        rows = []
        for f in files:
            gdf = gpd.read_file(f)
            if gdf.empty:
                continue
            if "issue" in gdf.columns:
                gdf["event_date"] = pd.to_datetime(gdf["issue"], errors="coerce", utc=True).dt.tz_localize(None)
            elif "polygon_begin" in gdf.columns:
                gdf["event_date"] = pd.to_datetime(gdf["polygon_begin"], errors="coerce", utc=True).dt.tz_localize(None)
            else:
                gdf["event_date"] = pd.NaT
            if "phenomena" not in gdf.columns:
                continue
            gdf = gdf[gdf["event_date"].notna()].copy()
            if gdf.empty:
                continue
            gdf["hazard"] = gdf["phenomena"].map(NOAA_HAZARD_MAP).fillna("other")
            rows.append(gdf[["event_date", "hazard", "geometry"]])

        if not rows:
            return None

        all_gdf = pd.concat(rows, ignore_index=True)
        all_gdf["event_day"] = all_gdf["event_date"].dt.floor("D")
        day_index = pd.date_range(
            self.config.time.start_date,
            self.config.time.end_date,
            freq="D",
        )
        y = np.full((len(day_index), h, w), self.config.label.label_default_value, dtype="int16")
        day_to_i = {pd.Timestamp(d): i for i, d in enumerate(day_index)}

        priority = self.config.label.label_priority or []
        selected = self._selected_hazards()
        ordered_hazards = [h for h in priority if h in selected]
        remaining = [h for h in sorted(selected) if h not in ordered_hazards]
        hazard_order = ordered_hazards + remaining

        for day, grp_day in all_gdf.groupby("event_day"):
            i = day_to_i.get(pd.Timestamp(day))
            if i is None:
                continue
            cell = np.full((h, w), self.config.label.label_default_value, dtype="int16")
            for hazard in hazard_order:
                hazard_id = self._label_id(hazard)
                if hazard_id == self.config.label.label_default_value:
                    continue
                gsub = grp_day[grp_day["hazard"] == hazard]
                if gsub.empty:
                    continue
                geom = gsub.geometry.unary_union
                prepared = prep(geom)
                mask = np.array([prepared.contains(pt) for pt in points], dtype=bool).reshape(h, w)
                cell[(mask) & (cell == self.config.label.label_default_value)] = hazard_id
            y[i, :, :] = cell

        return Grid(
            data=y,
            coords={"time": day_index.values, "lat": grid["lat"], "lon": grid["lon"]},
            attrs={"label_source": "noaa"},
        )

    def _align_labels(self, label_grid: Optional[Grid], feature_grid: Grid) -> Tuple[Grid, np.ndarray]:
        target_index = pd.to_datetime(feature_grid.coords["time"])
        if label_grid is None:
            zeros = np.zeros((len(target_index), feature_grid.data.shape[-2], feature_grid.data.shape[-1]), dtype="int16")
            mask = np.ones_like(zeros, dtype=bool)
            return (
                Grid(
                    data=zeros,
                    coords=feature_grid.coords,
                    attrs={"label_rule": self.config.synthetic.label_method},
                ),
                mask,
            )

        method = self.config.synthetic.label_method if self.config.synthetic.synthetic_time else "nearest"
        aligned, synth_t = self._resample_grid(
            Grid(
                data=label_grid.data[:, None, :, :],
                coords=label_grid.coords,
                attrs=label_grid.attrs,
            ),
            target_index=target_index,
            method=method,
            source_kind="label",
        )
        y = aligned.data[:, 0, :, :].astype("int16")
        y_mask = np.repeat(synth_t[:, None, None], y.shape[1], axis=1)
        y_mask = np.repeat(y_mask, y.shape[2], axis=2)
        return Grid(data=y, coords=feature_grid.coords, attrs=label_grid.attrs), y_mask

    def build(self) -> Sample:
        bbox = self.config.grid.bbox
        firms_df: Optional[pd.DataFrame] = None
        mtbs_df: Optional[pd.DataFrame] = None
        if bbox is None:
            # Only load what is needed for bbox inference.
            infer_frames: List[pd.DataFrame] = []
            if "firms" in self.config.sources:
                firms_df = get_adapter("firms")().load_points(self.config.root_dir, self.config.time)
                infer_frames.append(firms_df)
            if "mtbs_fod" in self.config.sources:
                mtbs_df = get_adapter("mtbs_fod")(label_map={}).load_points(
                    self.config.root_dir, self.config.time
                )
                infer_frames.append(mtbs_df)
            bbox = self._ensure_bbox(infer_frames)

        features: List[Tuple[str, Grid]] = []
        for source in self.config.sources:
            if source == "mtbs_fod":
                continue

            if not self._source_maybe_available(source):
                self._warn_missing_source(source)
                continue

            grid: Optional[Grid] = None
            if source == "firms":
                if firms_df is None:
                    firms_df = get_adapter("firms")().load_points(self.config.root_dir, self.config.time)
                if not firms_df.empty:
                    grid = rasterize_points_daily(
                        firms_df,
                        bbox=bbox,
                        resolution=self.config.grid.resolution_deg,
                        value_columns=["frp"],
                    )
            elif source in ("era5", "merra2"):
                grid = get_adapter(source)().load(
                    self.config.root_dir,
                    self.config.time,
                    bbox=bbox,
                    resolution=self.config.grid.resolution_deg,
                )
            elif source == "noaa":
                grid = get_adapter("noaa")().load(
                    self.config.root_dir,
                    self.config.time,
                    bbox=bbox,
                    resolution=self.config.grid.resolution_deg,
                )
            elif source == "landfire":
                grid = get_adapter("landfire")().load(
                    self.config.root_dir,
                    bbox=bbox,
                    resolution=self.config.grid.resolution_deg,
                )

            if grid is None:
                if not self.config.synthetic.synthetic_time:
                    self._warn_missing_source(source)
                continue
            features.append((source, grid))

        if not features:
            self._warn_no_features()
            target_index = pd.to_datetime(self._timeline_from_config())
            g = make_grid(bbox, self.config.grid.resolution_deg)
            h, w = len(g["lat"]), len(g["lon"])
            x = np.zeros((len(target_index), 0, h, w), dtype="float32")
            y = np.zeros((len(target_index), h, w), dtype="int16")
            return Sample(
                x=x,
                y=y,
                meta={
                    "config": asdict(self.config),
                    "channels": [],
                    "x_synthetic_mask": np.zeros((len(target_index), 0), dtype=bool),
                    "y_synthetic_mask": np.ones((len(target_index), h, w), dtype=bool),
                },
            )

        if self.config.synthetic.synthetic_time:
            x_grid, x_synth_mask = self._align_synthetic(features)
        else:
            x_grid, x_synth_mask = self._align_by_intersection(features)

        src = self.config.label.label_source
        if src in ("mtbs", "mtbs_fod"):
            label_grid = self._load_mtbs_labels(bbox)
        elif src == "noaa":
            label_grid = self._load_noaa_labels(bbox)
        elif src == "firms":
            label_grid = self._load_firms_labels(bbox)
        else:
            label_grid = None
        if label_grid is None and src is not None and not self.config.synthetic.synthetic_time:
            self._warn_missing_labels(src)
        y_grid, y_synth_mask = self._align_labels(label_grid, x_grid)

        return Sample(
            x=x_grid.data,
            y=y_grid.data,
            meta={
                "config": asdict(self.config),
                "channels": x_grid.attrs.get("channels", []),
                "x_synthetic_mask": x_synth_mask,
                "y_synthetic_mask": y_synth_mask,
            },
        )
