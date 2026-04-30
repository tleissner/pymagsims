# src/pymagsims/interactive.py
# SPDX-License-Identifier: GPL-3.0-or-later

import plotly.graph_objects as go


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
        )
    )

    fig.update_layout(
        title=spectrum.name or "MagSIMS spectrum",
        xaxis_title=x,
        yaxis_title=y,
        template="plotly_white",
    )

    if log_y:
        fig.update_yaxes(type="log")

    if xlim is not None:
        fig.update_xaxes(range=list(xlim))

    return fig