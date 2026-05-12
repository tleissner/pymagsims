# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class SIMSDepthProfile:
    profiles: pd.DataFrame
    metadata: dict = field(default_factory=dict)
    roi_table: pd.DataFrame | None = None
    last_spectrum: object | None = None
    raw_histograms: pd.DataFrame | None = None
    name: str | None = None

    @classmethod
    def from_fpd_csv(cls, path: str | Path, **kwargs) -> "SIMSDepthProfile":
        from .io import read_fpd_depth_profile_csv
        return read_fpd_depth_profile_csv(path, **kwargs)

    @classmethod
    def from_fpd_raw(cls, path: str | Path, **kwargs) -> "SIMSDepthProfile":
        from .io import read_fpd_depth_profile_raw
        return read_fpd_depth_profile_raw(path, **kwargs)

    def labels(self) -> list[str]:
        return [
            col for col in self.profiles.columns
            if col not in {"Acquisition", "Acquisition point", "Image number", "Depth"}
        ]

    def plot(self, labels=None, log_y: bool = False, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        else:
            fig = ax.figure

        if labels is None:
            labels = self.labels()

        if "Depth" in self.profiles.columns:
            x_col = "Depth"
        elif "Acquisition point" in self.profiles.columns:
            x_col = "Acquisition point"
        elif "Image number" in self.profiles.columns:
            x_col = "Image number"
        else:
            x_col = "Acquisition"

        for label in labels:
            ax.plot(
                self.profiles[x_col],
                self.profiles[label],
                marker="o",
                linewidth=1,
                label=label,
            )

        ax.set_xlabel(x_col)
        ax.set_ylabel("Integrated counts")

        if log_y:
            ax.set_yscale("log")

        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        return fig, ax
    

    def spectrum_for_layer(self, layer: int, calibration):
        from .spectrum import Spectrum
        import pandas as pd

        if self.raw_histograms is None:
            raise ValueError("No raw_histograms stored.")

        row = self.raw_histograms.iloc[layer]

        data = pd.DataFrame({
            "Channel": calibration.data["Channel"].values[:len(row)],
            "Mass": calibration.data["Mass"].values[:len(row)],
            "Amplitude": row.values,
        })

        return Spectrum(
            data=data,
            metadata=self.metadata.copy(),
            roi_table=self.roi_table,
            name=f"{self.name}_layer_{layer}",
        )


    def merged_spectrum(self, calibration=None):
        from .spectrum import Spectrum
        import pandas as pd
        import numpy as np

        if self.raw_histograms is None:
            raise ValueError("No raw_histograms stored. Load from raw file first.")

        summed = self.raw_histograms.sum(axis=0)
        channels = np.arange(1, len(summed) + 1)

        if calibration is not None:
            n = min(len(summed), len(calibration.data))
            data = pd.DataFrame({
                "Channel": calibration.data["Channel"].values[:n],
                "Mass": calibration.data["Mass"].values[:n],
                "Amplitude": summed.values[:n],
            })
        else:
            data = pd.DataFrame({
                "Channel": channels,
                "Mass": np.nan,
                "Amplitude": summed.values,
            })

        return Spectrum(data=data, metadata=self.metadata.copy(), name=f"{self.name}_merged")