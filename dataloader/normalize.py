from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from dataloader.schema import Grid


def parse_date(value: str) -> Optional[datetime]:
    if value is None or value == "":
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(str(value), fmt)
        except ValueError:
            continue
    return None


def make_grid(bbox: Tuple[float, float, float, float], resolution: float) -> Dict[str, np.ndarray]:
    minx, miny, maxx, maxy = bbox
    lon_edges = np.arange(minx, maxx + resolution, resolution, dtype="float64")
    lat_edges = np.arange(miny, maxy + resolution, resolution, dtype="float64")
    lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2.0
    lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    return {
        "lon_edges": lon_edges,
        "lat_edges": lat_edges,
        "lon": lon_centers,
        "lat": lat_centers,
    }


def _date_strings(times: np.ndarray) -> List[str]:
    out: List[str] = []
    for t in times:
        try:
            dt = pd.to_datetime(t).date()
            out.append(dt.strftime("%Y-%m-%d"))
        except Exception:
            continue
    return out


def rasterize_polygons_daily(
    gdf: "pd.DataFrame",
    bbox: Tuple[float, float, float, float],
    resolution: float,
    date_column: str,
) -> Grid:
    """Rasterize polygon events into daily grids."""
    from shapely.geometry import Point
    from shapely.prepared import prep

    grid = make_grid(bbox, resolution)
    df = gdf.copy()
    df[date_column] = df[date_column].apply(lambda v: str(v) if v is not None else "")
    df["dt"] = pd.to_datetime(df[date_column], errors="coerce")
    df = df[df["dt"].notna()]

    dates = sorted({d.date() for d in df["dt"].tolist()})
    times = [datetime.combine(d, datetime.min.time()) for d in dates]
    h, w = len(grid["lat"]), len(grid["lon"])
    data = np.zeros((len(times), 1, h, w), dtype="float32")
    if not times:
        return Grid(data=data, coords={"time": np.array([]), "lat": grid["lat"], "lon": grid["lon"]}, attrs={"channels": ["noaa_flood"]})

    lon_centers = grid["lon"]
    lat_centers = grid["lat"]
    points = [Point(lon, lat) for lat in lat_centers for lon in lon_centers]

    time_index = {t.date(): i for i, t in enumerate(times)}
    for day, group in df.groupby(df["dt"].dt.date):
        if "geometry" not in group.columns:
            continue
        idx = time_index.get(day)
        if idx is None:
            continue
        geom = group.unary_union
        prepared = prep(geom)
        mask = np.array([prepared.contains(pt) for pt in points], dtype=bool).reshape(h, w)
        data[idx, 0, :, :] = mask.astype("float32")

    return Grid(
        data=data,
        coords={"time": np.array(times), "lat": grid["lat"], "lon": grid["lon"]},
        attrs={"channels": ["noaa_flood"]},
    )


def grid_from_xarray(
    ds: "xr.Dataset",
    bbox: Tuple[float, float, float, float],
    resolution: float,
    variables: List[str],
) -> Grid:
    """Convert an xarray Dataset into a Grid aligned to bbox/resolution."""
    import xarray as xr

    ds = ds.rename(
        {
            "valid_time": "time",
            "longitude": "lon",
            "latitude": "lat",
        }
    )
    minx, miny, maxx, maxy = bbox
    if float(ds.lat[0]) > float(ds.lat[-1]):
        ds = ds.sel(lat=slice(maxy, miny), lon=slice(minx, maxx))
    else:
        ds = ds.sel(lat=slice(miny, maxy), lon=slice(minx, maxx))

    ds = ds[variables]
    ds = ds.resample(time="1D").mean()

    grid = make_grid(bbox, resolution)
    ds = ds.interp(lat=grid["lat"], lon=grid["lon"])

    data = np.stack([ds[var].values for var in variables], axis=1)
    return Grid(
        data=data.astype("float32"),
        coords={"time": ds["time"].values, "lat": grid["lat"], "lon": grid["lon"]},
        attrs={"channels": variables},
    )


def filter_bbox(df: pd.DataFrame, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
    minx, miny, maxx, maxy = bbox
    return df[
        (df["longitude"] >= minx)
        & (df["longitude"] <= maxx)
        & (df["latitude"] >= miny)
        & (df["latitude"] <= maxy)
    ]


def _time_index(dates: Iterable[datetime]) -> List[datetime]:
    uniq = sorted({d.date() for d in dates if d is not None})
    return [datetime.combine(d, datetime.min.time()) for d in uniq]


def rasterize_points_daily(
    df: pd.DataFrame,
    bbox: Tuple[float, float, float, float],
    resolution: float,
    value_columns: Optional[List[str]] = None,
) -> Grid:
    """Rasterize point events into daily grids."""
    if value_columns is None:
        value_columns = []

    grid = make_grid(bbox, resolution)
    df = filter_bbox(df, bbox)
    df = df.copy()
    if "date" not in df.columns:
        raise KeyError("Expected a 'date' column with YYYY-MM-DD values.")
    df["date"] = df["date"].apply(parse_date)
    df = df[df["date"].notna()]

    times = _time_index(df["date"].tolist())
    if not times:
        data = np.zeros((0, 1, len(grid["lat"]), len(grid["lon"])), dtype="float32")
        return Grid(data=data, coords={"time": np.array([]), "lat": grid["lat"], "lon": grid["lon"]}, attrs={})

    t_index = {t.date(): i for i, t in enumerate(times)}
    h, w = len(grid["lat"]), len(grid["lon"])
    channels = ["count"] + value_columns
    data = np.zeros((len(times), len(channels), h, w), dtype="float32")

    lon_edges = grid["lon_edges"]
    lat_edges = grid["lat_edges"]

    for _, row in df.iterrows():
        lon = float(row["longitude"])
        lat = float(row["latitude"])
        t = row["date"].date()
        ti = t_index.get(t)
        if ti is None:
            continue
        x = np.searchsorted(lon_edges, lon, side="right") - 1
        y = np.searchsorted(lat_edges, lat, side="right") - 1
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        data[ti, 0, y, x] += 1.0
        for c_idx, col in enumerate(value_columns, start=1):
            val = row.get(col)
            if val is None or val == "":
                continue
            try:
                data[ti, c_idx, y, x] += float(val)
            except ValueError:
                continue

    return Grid(
        data=data,
        coords={"time": np.array(times), "lat": grid["lat"], "lon": grid["lon"]},
        attrs={"channels": channels},
    )


def rasterize_labels_daily(
    df: pd.DataFrame,
    bbox: Tuple[float, float, float, float],
    resolution: float,
    label_column: str,
    label_rule: str = "max",
) -> Grid:
    """Rasterize categorical labels into daily grids."""
    grid = make_grid(bbox, resolution)
    df = filter_bbox(df, bbox)
    df = df.copy()
    if "date" not in df.columns:
        raise KeyError("Expected a 'date' column with YYYY-MM-DD values.")
    df["date"] = df["date"].apply(parse_date)
    df = df[df["date"].notna()]

    times = _time_index(df["date"].tolist())
    h, w = len(grid["lat"]), len(grid["lon"])
    data = np.zeros((len(times), h, w), dtype="int16")

    if not times:
        return Grid(
            data=data,
            coords={"time": np.array([]), "lat": grid["lat"], "lon": grid["lon"]},
            attrs={"label_rule": label_rule},
        )

    t_index = {t.date(): i for i, t in enumerate(times)}
    lon_edges = grid["lon_edges"]
    lat_edges = grid["lat_edges"]

    for _, row in df.iterrows():
        lon = float(row["longitude"])
        lat = float(row["latitude"])
        t = row["date"].date()
        ti = t_index.get(t)
        if ti is None:
            continue
        x = np.searchsorted(lon_edges, lon, side="right") - 1
        y = np.searchsorted(lat_edges, lat, side="right") - 1
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        label = int(row[label_column])
        if label_rule == "max":
            data[ti, y, x] = max(data[ti, y, x], label)
        else:
            data[ti, y, x] = label

    return Grid(
        data=data,
        coords={"time": np.array(times), "lat": grid["lat"], "lon": grid["lon"]},
        attrs={"label_rule": label_rule},
    )
