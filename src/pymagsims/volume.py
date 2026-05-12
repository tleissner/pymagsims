# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import re

def natural_sort_key(path):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(path))
    ]

@dataclass
class SIMSVolume:
    volumes: dict[str, np.ndarray]
    metadata: dict = field(default_factory=dict)
    bins: pd.DataFrame | None = None
    name: str | None = None

    @classmethod
    def from_fpd_raw_image_series(
        cls,
        paths,
        spectrum,
        bins,
        include_total: bool = True,
        shape: tuple[int, int] | None = None,
    ) -> "SIMSVolume":
        from .raw_image import SIMSRawImage

        
        paths = sorted([Path(p) for p in paths], key=natural_sort_key)

        stacks = {}

        if include_total:
            stacks["Total"] = []

        for _, b in bins.iterrows():
            stacks[str(b["label"])] = []

        for path in paths:
            raw = SIMSRawImage.from_fpd_raw(path, shape=shape)

            if include_total:
                stacks["Total"].append(raw.total_ion_image())

            images = raw.images_from_bins(
                spectrum=spectrum,
                bins=bins,
                include_total=False,
            )

            for label, img in images.items():
                stacks[label].append(img)

        volumes = {
            label: np.stack(imgs, axis=0)
            for label, imgs in stacks.items()
        }

        return cls(
            volumes=volumes,
            metadata={
                "source": "FPD 3D raw image series",
                "n_slices": len(paths),
                "files": [str(p) for p in paths],
                "shape": {
                    label: vol.shape
                    for label, vol in volumes.items()
                },
            },
            bins=bins.copy(),
            name="FPD 3D volume",
        )

    def labels(self) -> list[str]:
        return list(self.volumes.keys())

    def get(self, label: str) -> np.ndarray:
        return self.volumes[label]

    def sum_projection(self, label: str) -> np.ndarray:
        return self.volumes[label].sum(axis=0)

    def max_projection(self, label: str) -> np.ndarray:
        return self.volumes[label].max(axis=0)

    def depth_profile(self, label: str) -> pd.DataFrame:
        vol = self.get(label)

        return pd.DataFrame(
            {
                "Slice": np.arange(vol.shape[0]),
                label: vol.sum(axis=(1, 2)),
            }
        )


    
    def merged_spectrum(self, label: str = "Total"):
        """
        Return depth-integrated intensity for one volume label.

        For now this works for already binned volumes, not full raw histograms.
        """
        vol = self.get(label)

        return vol.sum(axis=0)