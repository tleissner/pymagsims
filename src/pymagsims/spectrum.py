# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import pandas as pd


@dataclass
class Spectrum:
    data: pd.DataFrame
    metadata: dict = field(default_factory=dict)
    roi_table: pd.DataFrame | None = None
    raw_events: pd.DataFrame | None = None
    name: str | None = None

   
    @classmethod
    def from_main_analysis_file(cls, path: str | Path, **kwargs) -> Self:
        from .io import read_main_analysis_file
        return read_main_analysis_file(path, **kwargs)

    @classmethod
    def from_raw_file(cls, path: str | Path, **kwargs) -> Self:
        from .io import read_raw_file
        return read_raw_file(path, **kwargs)

    def plot(self, *args, **kwargs):
        from .plotting import plot_spectrum
        return plot_spectrum(self, *args, **kwargs)

    def plot_channel_histogram(self, *args, **kwargs):
        from .plotting import plot_channel_histogram
        return plot_channel_histogram(self, *args, **kwargs)

    def copy(self) -> "Spectrum":
        return Spectrum(
            data=self.data.copy(),
            metadata=self.metadata.copy(),
            roi_table=None if self.roi_table is None else self.roi_table.copy(),
            raw_events=None if self.raw_events is None else self.raw_events.copy(),
            name=self.name,
        )

    def find_peaks(self, *args, **kwargs):
        from .peaks import find_spectrum_peaks
        return find_spectrum_peaks(self, *args, **kwargs)

    def assign_peaks(self, *args, **kwargs):
        from .peaks import assign_spectrum_peaks
        return assign_spectrum_peaks(self, *args, **kwargs)

    def plot_with_peaks(self, *args, **kwargs):
        from .plotting import plot_spectrum_with_peaks
        return plot_spectrum_with_peaks(self, *args, **kwargs)


    def plot_with_element_markers(self, *args, **kwargs):
        from .plotting import plot_spectrum_with_element_markers
        return plot_spectrum_with_element_markers(self, *args, **kwargs)
    
    def plot_interactive(self, *args, **kwargs):
        from .interactive import plot_spectrum_interactive
        return plot_spectrum_interactive(self, *args, **kwargs)


    def plot_with_peaks_interactive(self, *args, **kwargs):
        from .interactive import plot_spectrum_with_peaks_interactive
        return plot_spectrum_with_peaks_interactive(self, *args, **kwargs)


    def plot_with_element_markers_interactive(self, *args, **kwargs):
        from .interactive import plot_spectrum_with_element_markers_interactive
        return plot_spectrum_with_element_markers_interactive(self, *args, **kwargs)
    

    def create_bins_from_assignments(self, assignments, *args, **kwargs):
        from .binning import create_bins_from_assignments
        return create_bins_from_assignments(assignments, spectrum=self, *args, **kwargs)