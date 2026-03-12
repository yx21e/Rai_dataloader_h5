from __future__ import annotations

import os
from typing import Optional

import numpy as np

from dataloader.schema import Grid


def _safe_path(cache_dir: str, name: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, name)


def save_grid(grid: Grid, cache_dir: str, name: str) -> str:
    path = _safe_path(cache_dir, f"{name}.npz")
    np.savez_compressed(path, data=grid.data, **grid.coords, **grid.attrs)
    return path


def load_grid(cache_dir: str, name: str) -> Optional[Grid]:
    path = os.path.join(cache_dir, f"{name}.npz")
    if not os.path.exists(path):
        return None
    payload = np.load(path, allow_pickle=True)
    coords = {k: payload[k] for k in ("time", "lat", "lon") if k in payload}
    attrs = {}
    for k in payload.files:
        if k in ("data", "time", "lat", "lon"):
            continue
        v = payload[k]
        # Scalars come back as 0-d arrays, lists/channels come back as 1-d arrays.
        if np.ndim(v) == 0:
            attrs[k] = v.item()
        else:
            attrs[k] = v.tolist()
    return Grid(data=payload["data"], coords=coords, attrs=attrs)
