# src/pymagsims/raw_image.py
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SIMSRawImage:
    events: pd.DataFrame
    metadata: dict = field(default_factory=dict)
    name: str | None = None

    @classmethod
    def from_fpd_raw(cls, path: str | Path, encoding: str = "latin1") -> "SIMSRawImage":
        path = Path(path)

        records = []

        with path.open("r", encoding=encoding) as f:
            for line in f:
                parts = [p for p in line.strip().split(";") if p != ""]

                if len(parts) < 2:
                    continue

                x = int(float(parts[0]))
                y = int(float(parts[1]))

                for channel in parts[2:]:
                    records.append((x, y, int(float(channel))))

        events = pd.DataFrame(records, columns=["X", "Y", "Channel"])

        metadata = {
            "source": "FPD raw image",
            "n_pixels_with_rows": int(events[["X", "Y"]].drop_duplicates().shape[0]),
            "n_events": int(len(events)),
            "x_min": int(events["X"].min()),
            "x_max": int(events["X"].max()),
            "y_min": int(events["Y"].min()),
            "y_max": int(events["Y"].max()),
            "channel_min": int(events["Channel"].min()),
            "channel_max": int(events["Channel"].max()),
        }

        return cls(
            events=events,
            metadata=metadata,
            name=path.stem,
        )

    @property
    def shape(self) -> tuple[int, int]:
        """
        Return image shape as (height, width).
        """
        height = int(self.events["Y"].max() - self.events["Y"].min() + 1)
        width = int(self.events["X"].max() - self.events["X"].min() + 1)
        return height, width

    def total_ion_image(self) -> np.ndarray:
        """
        Create total ion image from all events.
        """

        y_min = int(self.events["Y"].min())
        x_min = int(self.events["X"].min())

        img = np.zeros(self.shape, dtype=np.int32)

        yy = self.events["Y"].to_numpy(dtype=int) - y_min
        xx = self.events["X"].to_numpy(dtype=int) - x_min

        np.add.at(img, (yy, xx), 1)

        return img

    def image_from_channel_bin(
        self,
        ch_min: int,
        ch_max: int,
    ) -> np.ndarray:
        """
        Create ion image from a detector-channel bin.
        """

        subset = self.events[
            (self.events["Channel"] >= ch_min)
            & (self.events["Channel"] <= ch_max)
        ]

        y_min = int(self.events["Y"].min())
        x_min = int(self.events["X"].min())

        img = np.zeros(self.shape, dtype=np.int32)

        yy = subset["Y"].to_numpy(dtype=int) - y_min
        xx = subset["X"].to_numpy(dtype=int) - x_min

        np.add.at(img, (yy, xx), 1)

        return img
    
    def image_from_mass_bin(
        self,
        spectrum,
        mass_min: float,
        mass_max: float,
        ) -> np.ndarray:
            """
            Create ion image from a mass bin using a Spectrum calibration.
            """

            ch_min = mass_to_channel(spectrum, mass_min)
            ch_max = mass_to_channel(spectrum, mass_max)

            if ch_max < ch_min:
                ch_min, ch_max = ch_max, ch_min

            return self.image_from_channel_bin(ch_min, ch_max)


    def images_from_bins(
        self,
        spectrum,
        bins,
        include_total: bool = True,
    ) -> dict[str, np.ndarray]:
        images = {}

        if include_total:
            images["Total"] = self.total_ion_image()

        for _, b in bins.iterrows():
            label = str(b["label"])

            images[label] = self.image_from_mass_bin(
                spectrum=spectrum,
                mass_min=float(b["mass_min"]),
                mass_max=float(b["mass_max"]),
            )

        return images
    
def mass_to_channel(spectrum, mass: float) -> int:
    """
    Convert mass to nearest detector channel using Spectrum calibration.
    """

    idx = (spectrum.data["Mass"] - mass).abs().idxmin()
    return int(spectrum.data.loc[idx, "Channel"])