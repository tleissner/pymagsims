# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_channel_histogram(spectrum, log_y: bool = False, ax=None):
    return plot_spectrum(
        spectrum,
        x="Channel",
        y="Amplitude",
        log_y=log_y,
        ax=ax,
    )


def plot_spectrum(
    spectrum,
    x: str = "Mass",
    y: str = "Amplitude",
    log_y: bool = False,
    xlim: tuple[float, float] | None = None,
    ax=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        fig = ax.figure

    ax.plot(spectrum.data[x], spectrum.data[y], linewidth=1)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(spectrum.name or "MagSIMS spectrum")

    if log_y:
        ax.set_yscale("log")

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.grid(True)
    fig.tight_layout()
    return fig, ax


def plot_spectrum_with_peaks(
    spectrum,
    isotope_table,
    tolerance: float = 0.2,
    prominence: float | None = 100,
    height: float | None = None,
    distance: int | None = 5,
    log_y: bool = False,
    annotate: bool = True,
    xlim: tuple[float, float] | None = None,
    ax=None,
):
    """
    Plot spectrum with detected and isotope-assigned peaks.

    Shows:
    - measured spectrum
    - detected peak markers
    - isotope labels for assigned peaks
    """

    assignments = spectrum.assign_peaks(
        isotope_table=isotope_table,
        tolerance=tolerance,
        prominence=prominence,
        height=height,
        distance=distance,
    )

    peaks = spectrum.find_peaks(
        prominence=prominence,
        height=height,
        distance=distance,
    )

    fig, ax = plot_spectrum(spectrum, log_y=log_y, ax=ax)

    ax.scatter(
        peaks["measured_mass"],
        peaks["intensity"],
        marker="x",
        label="Detected peaks",
    )

    if annotate and not assignments.empty:
        best_assignments = (
            assignments.sort_values("abs_mass_error")
            .drop_duplicates("measured_mass")
        )

        for _, row in best_assignments.iterrows():
            ax.text(
                row["measured_mass"],
                row["intensity"],
                row["isotope"],
                rotation=90,
                fontsize=8,
                va="bottom",
                ha="center",
            )
            
    if xlim is not None:
        ax.set_xlim(xlim)
        
    ax.legend()
    fig.tight_layout()

    return fig, ax, assignments


def plot_spectrum_with_element_markers(
    spectrum,
    isotope_table,
    elements: list[str],
    log_y: bool = False,
    min_abundance: float = 0.0,
    mass_min: float | None = None,
    mass_max: float | None = None,
    xlim: tuple[float, float] | None = None,
    ax=None,
):
    """
    Plot measured spectrum plus theoretical isotope positions
    for up to three selected elements.

    Example:
        spec.plot_with_element_markers(
            isotopes,
            elements=["Si", "Ga", "O"],
            log_y=True,
        )
    """

    if len(elements) > 3:
        raise ValueError("Please provide at most three elements.")

    fig, ax = plot_spectrum(spectrum, log_y=log_y, ax=ax)

    if mass_min is None:
        mass_min = spectrum.data["Mass"].min()

    if mass_max is None:
        mass_max = spectrum.data["Mass"].max()

    ymax = spectrum.data["Amplitude"].max()

    for element in elements:
        element_isotopes = isotope_table[
            (isotope_table["element"] == element)
            & (isotope_table["abundance"] >= min_abundance)
            & (isotope_table["exact_mass"] >= mass_min)
            & (isotope_table["exact_mass"] <= mass_max)
        ]

        for _, iso in element_isotopes.iterrows():
            rel_height = iso["abundance"] / 100.0
            marker_height = ymax * max(rel_height, 0.05)

            ax.vlines(
                iso["exact_mass"],
                ymin=0,
                ymax=marker_height,
                linestyle="--",
                linewidth=1,
                label=element if iso.name == element_isotopes.index[0] else None,
            )

            ax.text(
                iso["exact_mass"],
                marker_height,
                iso["isotope"],
                rotation=90,
                fontsize=8,
                va="bottom",
                ha="center",
            )

    ax.set_title(
        f"{spectrum.name or 'MagSIMS spectrum'} with isotope markers"
    )

    if xlim is not None:
        ax.set_xlim(xlim)

        
    ax.legend()
    fig.tight_layout()

    return fig, ax