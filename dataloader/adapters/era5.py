from __future__ import annotations

import glob
import os
import re
from typing import List, Optional

import xarray as xr

from dataloader.config import TimeConfig
from dataloader.normalize import grid_from_xarray
from dataloader.schema import Grid


class ERA5Adapter:
    """Load ERA5 NetCDF data and regrid to target bbox/resolution."""

    DATA_DIRS = [
        "era5/era5_data",
        "era5/era510",
        "era5/era59",
        "era5/era5new",
    ]

    DEFAULT_VARS = ["t2m", "d2m", "u10", "v10", "sp", "swvl1"]

    def _list_files(self, root_dir: str) -> List[str]:
        files: List[str] = []
        for d in self.DATA_DIRS:
            base = os.path.join(root_dir, d)
            files.extend(glob.glob(os.path.join(base, "era5_*.nc")))
        return sorted(set(files))

    def _select_files(self, files: List[str], time_cfg: TimeConfig) -> List[str]:
        if not time_cfg.start_date and not time_cfg.end_date:
            return files
        start = time_cfg.start_date or "1900-01-01"
        end = time_cfg.end_date or "2999-12-31"
        sy, sm = int(start[:4]), int(start[5:7])
        ey, em = int(end[:4]), int(end[5:7])

        selected = []
        for f in files:
            m = re.search(r"era5_(\d{4})_(\d{2})", os.path.basename(f))
            if not m:
                continue
            y, mo = int(m.group(1)), int(m.group(2))
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
        variables: Optional[List[str]] = None,
    ) -> Optional[Grid]:
        files = self._select_files(self._list_files(root_dir), time_cfg)
        if not files:
            return None

        vars_req = variables or self.DEFAULT_VARS
        ds = xr.open_mfdataset(files, combine="by_coords")
        available = [v for v in vars_req if v in ds.data_vars]
        if not available:
            return None

        grid = grid_from_xarray(ds, bbox=bbox, resolution=resolution, variables=available)
        return grid
