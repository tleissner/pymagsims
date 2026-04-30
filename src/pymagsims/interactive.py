# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, clear_output
from dataclasses import dataclass, field



@dataclass
class ManualPeakBins:
    bins: list[dict] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.bins)

    def clear(self):
        self.bins.clear()

    def save_csv(self, path):
        self.to_dataframe().to_csv(path, index=False)


def plot_spectrum_interactive(
    spectrum,
    x: str = "Mass",
    y: str = "Amplitude",
    log_y: bool = False,
    xlim: tuple[float, float] | None = None,
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=spectrum.data[x],
            y=spectrum.data[y],
            mode="lines",
            name="Spectrum",
            hovertemplate=f"{x}: %{{x:.4f}}<br>{y}: %{{y}}<extra></extra>",
        )
    )

    fig.update_layout(
        title=spectrum.name or "MagSIMS spectrum",
        xaxis_title=x,
        yaxis_title=y,
        template="plotly_white",
        hovermode="closest",
    )

    if log_y:
        fig.update_yaxes(type="log")

    if xlim is not None:
        fig.update_xaxes(range=list(xlim))

    return fig


def plot_spectrum_with_peaks_interactive(
    spectrum,
    isotope_table,
    tolerance: float = 0.2,
    prominence: float | None = 100,
    height: float | None = None,
    distance: int | None = 5,
    log_y: bool = False,
    xlim: tuple[float, float] | None = None,
    label_peaks: bool = True,
):
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

    fig = plot_spectrum_interactive(
        spectrum,
        log_y=log_y,
        xlim=xlim,
    )

    fig.add_trace(
        go.Scatter(
            x=peaks["measured_mass"],
            y=peaks["intensity"],
            mode="markers",
            marker=dict(symbol="x", size=8),
            name="Detected peaks",
            hovertemplate=(
                "Mass: %{x:.4f}<br>"
                "Intensity: %{y}<br>"
                "<extra>Detected peak</extra>"
            ),
        )
    )

    if label_peaks and not assignments.empty:
        best = (
            assignments.sort_values("abs_mass_error")
            .drop_duplicates("measured_mass")
            .copy()
        )

        fig.add_trace(
            go.Scatter(
                x=best["measured_mass"],
                y=best["intensity"],
                mode="text",
                text=best["isotope"],
                textposition="top center",
                name="Peak labels",
                hovertemplate=(
                    "Measured mass: %{x:.4f}<br>"
                    "Assignment: %{text}<br>"
                    "<extra></extra>"
                ),
            )
        )

    return fig, assignments


def plot_spectrum_with_element_markers_interactive(
    spectrum,
    isotope_table,
    elements: list[str],
    log_y: bool = False,
    xlim: tuple[float, float] | None = None,
    min_abundance: float = 0.0,
):
    if len(elements) > 3:
        raise ValueError("Please provide at most three elements.")

    fig = plot_spectrum_interactive(
        spectrum,
        log_y=log_y,
        xlim=xlim,
    )

    mass_min = spectrum.data["Mass"].min() if xlim is None else xlim[0]
    mass_max = spectrum.data["Mass"].max() if xlim is None else xlim[1]

    ymax = spectrum.data["Amplitude"].max()

    for element in elements:
        element_isotopes = isotope_table[
            (isotope_table["element"] == element)
            & (isotope_table["abundance"] >= min_abundance)
            & (isotope_table["exact_mass"] >= mass_min)
            & (isotope_table["exact_mass"] <= mass_max)
        ].copy()

        for _, iso in element_isotopes.iterrows():
            marker_height = ymax * max(float(iso["abundance"]) / 100.0, 0.05)

            fig.add_trace(
                go.Scatter(
                    x=[iso["exact_mass"], iso["exact_mass"]],
                    y=[1 if log_y else 0, marker_height],
                    mode="lines+text",
                    text=["", iso["isotope"]],
                    textposition="top center",
                    line=dict(dash="dash"),
                    name=element,
                    hovertemplate=(
                        f"Element: {element}<br>"
                        f"Isotope: {iso['isotope']}<br>"
                        "Exact mass: %{x:.5f}<br>"
                        f"Abundance: {iso['abundance']} %"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=f"{spectrum.name or 'MagSIMS spectrum'} with isotope markers"
    )

    return fig

def manual_peak_binner_interactive(spectrum, log_y=False, xlim=None):
    """Create an interactive manual peak binning widget for spectrum analysis."""
    df = spectrum.data.copy()

    roi_min = widgets.FloatText(description="Mass min")
    roi_max = widgets.FloatText(description="Mass max")
    label = widgets.Text(description="Label")
    add_button = widgets.Button(description="Add ROI/bin")
    output = widgets.Output()

    result = ManualPeakBins()

    def make_fig():
        fig = go.Figure()
        fig.add_scatter(
            x=df["Mass"],
            y=df["Amplitude"],
            mode="lines",
            name="Spectrum",
        )

        for b in result.bins:
            fig.add_vrect(
                x0=b["mass_min"],
                x1=b["mass_max"],
                fillcolor="LightSalmon",
                opacity=0.25,
                line_width=0,
                annotation_text=b["label"],
                annotation_position="top left",
            )

        fig.update_layout(
            title="Manual peak binning",
            xaxis_title="Mass",
            yaxis_title="Amplitude",
            template="plotly_white",
        )

        if log_y:
            fig.update_yaxes(type="log")

        if xlim is not None:
            fig.update_xaxes(range=list(xlim))

        return fig

    fig_out = widgets.Output()

    def redraw():
        with fig_out:
            clear_output(wait=True)
            display(make_fig())

    def add_roi(_):
        x0 = float(roi_min.value)
        x1 = float(roi_max.value)

        if x1 < x0:
            x0, x1 = x1, x0

        roi = df[(df["Mass"] >= x0) & (df["Mass"] <= x1)]

        if roi.empty:
            with output:
                clear_output(wait=True)
                print(f"No data in ROI {x0:.4f}–{x1:.4f}")
            return

        max_row = roi.loc[roi["Amplitude"].idxmax()]

        result.bins.append(
            {
                "label": label.value,
                "mass_min": x0,
                "mass_max": x1,
                "center_mass": float(max_row["Mass"]),
                "max_intensity": float(max_row["Amplitude"]),
                "integrated_intensity": float(roi["Amplitude"].sum()),
                "n_points": int(len(roi)),
            }
        )

        redraw()

        with output:
            clear_output(wait=True)
            display(result.to_dataframe())

    add_button.on_click(add_roi)

    controls = widgets.HBox([roi_min, roi_max, label, add_button])
    display(widgets.VBox([controls, fig_out, output]))

    redraw()

    return result