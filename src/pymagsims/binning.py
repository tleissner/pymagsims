# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import pandas as pd


def create_bins_from_assignments(
    assignments: pd.DataFrame,
    spectrum=None,
    width: float | None = None,
    width_factor: float = 1.5,
    min_width: float = 0.15,
    max_width: float = 0.6,
    merge_overlapping: bool = True,
) -> pd.DataFrame:
    """
    Create mass-integration bins from identified/labeled peaks.

    Parameters
    ----------
    assignments:
        Output from spectrum.assign_peaks(...).
    spectrum:
        Optional Spectrum object. If provided, bins get integrated intensities.
    width:
        Full bin width in amu. If None, width is estimated from assignment FWHM
        if available, otherwise min_width is used.
    width_factor:
        Multiplier applied to FWHM when available.
    min_width:
        Minimum full bin width in amu.
    max_width:
        Maximum full bin width in amu.
    merge_overlapping:
        Merge overlapping bins belonging to the same label.

    Returns
    -------
    pd.DataFrame with columns:
        label, element, isotope, center_mass, mass_min, mass_max,
        integrated_intensity, max_intensity, n_points
    """

    if assignments.empty:
        return pd.DataFrame(
            columns=[
                "label",
                "element",
                "isotope",
                "center_mass",
                "mass_min",
                "mass_max",
                "integrated_intensity",
                "max_intensity",
                "n_points",
            ]
        )

    best = (
        assignments.sort_values("abs_mass_error")
        .drop_duplicates("measured_mass")
        .copy()
    )

    bins = []

    for _, row in best.iterrows():
        center = float(row["measured_mass"])

        if width is not None:
            bin_width = float(width)
        elif "widths" in row and pd.notna(row["widths"]):
            # scipy width is in index/channel units, so only use cautiously.
            bin_width = min_width
        else:
            bin_width = min_width

        bin_width = max(min_width, min(max_width, bin_width * width_factor))
        half_width = bin_width / 2

        label = str(row.get("isotope", row.get("element", "")))

        entry = {
            "label": label,
            "element": row.get("element", None),
            "isotope": row.get("isotope", None),
            "center_mass": center,
            "exact_mass": row.get("exact_mass", None),
            "mass_error": row.get("mass_error", None),
            "mass_min": center - half_width,
            "mass_max": center + half_width,
        }

        bins.append(entry)

    bins_df = pd.DataFrame(bins)

    if merge_overlapping:
        bins_df = merge_overlapping_bins(bins_df)

    if spectrum is not None:
        bins_df = integrate_bins(spectrum, bins_df)

    return bins_df.reset_index(drop=True)


def integrate_bins(spectrum, bins: pd.DataFrame) -> pd.DataFrame:
    """
    Add integrated intensity information to bins.
    """

    df = spectrum.data
    out = bins.copy()

    integrated = []
    max_intensity = []
    n_points = []

    for _, b in out.iterrows():
        roi = df[
            (df["Mass"] >= b["mass_min"])
            & (df["Mass"] <= b["mass_max"])
        ]

        integrated.append(float(roi["Amplitude"].sum()))

        if roi.empty:
            max_intensity.append(0.0)
            n_points.append(0)
        else:
            max_intensity.append(float(roi["Amplitude"].max()))
            n_points.append(int(len(roi)))

    out["integrated_intensity"] = integrated
    out["max_intensity"] = max_intensity
    out["n_points"] = n_points

    return out


def merge_overlapping_bins(bins: pd.DataFrame) -> pd.DataFrame:
    """
    Merge overlapping bins with the same label.
    """

    if bins.empty:
        return bins

    merged = []

    for label, group in bins.sort_values("mass_min").groupby("label"):
        current = None

        for _, row in group.iterrows():
            row = row.to_dict()

            if current is None:
                current = row
                continue

            if row["mass_min"] <= current["mass_max"]:
                current["mass_max"] = max(current["mass_max"], row["mass_max"])
                current["center_mass"] = (
                    current["mass_min"] + current["mass_max"]
                ) / 2
            else:
                merged.append(current)
                current = row

        if current is not None:
            merged.append(current)

    return pd.DataFrame(merged)