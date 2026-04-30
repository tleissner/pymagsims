# src/pymagsims/image.py
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SIMSImage:
    images: dict[str, np.ndarray]
    metadata: dict = field(default_factory=dict)
    roi_table: pd.DataFrame | None = None
    spectrum = None
    name: str | None = None

    @classmethod
    def from_fpd_csv(cls, path: str | Path, **kwargs) -> "SIMSImage":
        from .io import read_fpd_image_csv
        return read_fpd_image_csv(path, **kwargs)

    def labels(self) -> list[str]:
        return list(self.images.keys())

    def get(self, label: str) -> np.ndarray:
        return self.images[label]

    def total(self) -> np.ndarray:
        if "Total" not in self.images:
            raise KeyError("No 'Total' image found.")
        return self.images["Total"]