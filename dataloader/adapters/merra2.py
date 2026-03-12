from __future__ import annotations

import glob
import os
from typing import List, Optional

import xarray as xr

from dataloader.config import TimeConfig
from dataloader.normalize import grid_from_xarray
from dataloader.schema import Grid


class MERRA2Adapter:
    """Load MERRA2 NetCDF data and regrid to target bbox/resolution."""

    DATA_DIR = "merra2"
    DEFAULT_VARS = ["T2M", "U10M", "V10M", "PS", "QV2M"]

    def _list_files(self, root_dir: str) -> List[str]:
        base = os.path.join(root_dir, self.DATA_DIR)
        return sorted(glob.glob(os.path.join(base, "**", "*.nc"), recursive=True))

    def load(
        self,
        root_dir: str,
        time_cfg: TimeConfig,
        bbox: tuple[float, float, float, float],
        resolution: float,
        variables: Optional[List[str]] = None,
    ) -> Optional[Grid]:
        files = self._list_files(root_dir)
        if not files:
            return None

        vars_req = variables or self.DEFAULT_VARS
        ds = xr.open_mfdataset(files, combine="by_coords")
        available = [v for v in vars_req if v in ds.data_vars]
        if not available:
            return None

        grid = grid_from_xarray(ds, bbox=bbox, resolution=resolution, variables=available)
        return grid
