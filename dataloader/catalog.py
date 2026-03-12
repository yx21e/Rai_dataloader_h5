from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class Asset:
    source: str
    path: str


def discover_files(root: str, patterns: Iterable[str]) -> List[str]:
    hits: List[str] = []
    for pattern in patterns:
        hits.extend(glob.glob(os.path.join(root, pattern)))
        hits.extend(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    return sorted(set(hits))


def build_catalog(root: str, source: str, patterns: Iterable[str]) -> List[Asset]:
    return [Asset(source=source, path=p) for p in discover_files(root, patterns)]


def pick_first(paths: Iterable[str]) -> Optional[str]:
    for path in paths:
        return path
    return None
