# src/pymagsims/peaks.py
# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import pandas as pd
from scipy.signal import find_peaks


def find_spectrum_peaks(
    spectrum,
    height: float | None = None,
    prominence: float | None = None,
    distance: int | None = None,
    width: float | None = None,
    mass_column: str = "Mass",
    intensity_column: str = "Amplitude",
) -> pd.DataFrame:
    """
    Detect peaks in a MagSIMS spectrum.

    Returns a DataFrame with peak position, intensity, and scipy peak properties.
    """

    y = spectrum.data[intensity_column].to_numpy()

    peak_indices, properties = find_peaks(
        y,
        height=height,
        prominence=prominence,
        distance=distance,
        width=width,
    )

    peaks = spectrum.data.iloc[peak_indices].copy()
    peaks["peak_index"] = peak_indices

    for key, value in properties.items():
        peaks[key] = value

    peaks = peaks.rename(
        columns={
            mass_column: "measured_mass",
            intensity_column: "intensity",
        }
    )

    return peaks.reset_index(drop=True)


def assign_spectrum_peaks(
    spectrum,
    isotope_table: pd.DataFrame,
    tolerance: float = 0.2,
    height: float | None = None,
    prominence: float | None = None,
    distance: int | None = None,
) -> pd.DataFrame:
    """
    Detect peaks and assign possible isotopes.

    isotope_table must contain:
        element
        isotope
        exact_mass
        abundance
    """

    peaks = find_spectrum_peaks(
        spectrum,
        height=height,
        prominence=prominence,
        distance=distance,
    )

    return assign_peaks_to_isotopes(
        peaks=peaks,
        isotope_table=isotope_table,
        tolerance=tolerance,
    )


def assign_peaks_to_isotopes(
    peaks: pd.DataFrame,
    isotope_table: pd.DataFrame,
    tolerance: float = 0.2,
) -> pd.DataFrame:
    assignments = []

    for _, peak in peaks.iterrows():
        measured_mass = peak["measured_mass"]

        candidates = isotope_table[
            isotope_table["exact_mass"].between(
                measured_mass - tolerance,
                measured_mass + tolerance,
            )
        ].copy()

        for _, candidate in candidates.iterrows():
            mass_error = measured_mass - candidate["exact_mass"]

            assignments.append(
                {
                    "measured_mass": measured_mass,
                    "intensity": peak["intensity"],
                    "element": candidate["element"],
                    "isotope": candidate["isotope"],
                    "exact_mass": candidate["exact_mass"],
                    "abundance": candidate["abundance"],
                    "mass_error": mass_error,
                    "abs_mass_error": abs(mass_error),
                }
            )

    if not assignments:
        return pd.DataFrame(
            columns=[
                "measured_mass",
                "intensity",
                "element",
                "isotope",
                "exact_mass",
                "abundance",
                "mass_error",
                "abs_mass_error",
            ]
        )

    return (
        pd.DataFrame(assignments)
        .sort_values(["measured_mass", "abs_mass_error"])
        .reset_index(drop=True)
    )

