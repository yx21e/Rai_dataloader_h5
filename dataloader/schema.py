from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class Grid:
    """Simple grid container."""

    data: np.ndarray
    coords: Dict[str, np.ndarray]
    attrs: Dict[str, Any]

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape


@dataclass
class Sample:
    """Unified sample output."""

    x: np.ndarray
    y: np.ndarray
    meta: Dict[str, Any]
