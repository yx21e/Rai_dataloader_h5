from __future__ import annotations

from typing import Dict, Type

from dataloader.adapters.era5 import ERA5Adapter
from dataloader.adapters.firms import FIRMSAdapter
from dataloader.adapters.landfire import LandfireAdapter
from dataloader.adapters.merra2 import MERRA2Adapter
from dataloader.adapters.mtbs_fod import MTBSFODAdapter
from dataloader.adapters.noaa import NOAAAdapter


ADAPTERS: Dict[str, Type] = {
    "firms": FIRMSAdapter,
    "mtbs_fod": MTBSFODAdapter,
    "era5": ERA5Adapter,
    "merra2": MERRA2Adapter,
    "noaa": NOAAAdapter,
    "landfire": LandfireAdapter,
}


def get_adapter(name: str):
    if name not in ADAPTERS:
        raise KeyError(f"Unknown adapter: {name}")
    return ADAPTERS[name]
