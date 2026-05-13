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

def plot_ion_image(
    image,
    log: bool = True,
    title: str | None = None,
    cmap: str = "viridis",
    ax=None,
):
    import numpy as np
    import matplotlib.pyplot as plt

    data = np.log1p(image) if log else image

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    im = ax.imshow(data, cmap=cmap)
    ax.set_title(title or "Ion image")
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")

    fig.colorbar(im, ax=ax, label="log(1 + counts)" if log else "counts")
    fig.tight_layout()

    return fig, ax

def plot_ion_image_grid(
    images: dict,
    log: bool = True,
    ncols: int = 3,
    cmaps: list[str] | None = None,
    figsize_per_panel: tuple[float, float] = (4, 4),
):
    import math
    import numpy as np
    import matplotlib.pyplot as plt

    if cmaps is None:
        cmaps = [
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "turbo",
            "Greys",
            "hot",
            "cool",
            "spring",
        ]

    n_images = len(images)
    nrows = math.ceil(n_images / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    for ax in axes.ravel():
        ax.axis("off")

    for i, (label, image) in enumerate(images.items()):
        ax = axes.ravel()[i]

        data = np.log1p(image) if log else image
        cmap = cmaps[i % len(cmaps)]

        im = ax.imshow(data, cmap=cmap)
        ax.set_title(str(label))
        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")
        ax.axis("on")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("log(1 + counts)" if log else "counts")

    fig.tight_layout()
    return fig, axes


def plot_volume_slice(
    volume,
    label: str,
    z: int,
    log: bool = True,
    cmap: str = "viridis",
):
    import numpy as np
    import matplotlib.pyplot as plt

    arr = volume.get(label)
    img = arr[z]
    data = np.log1p(img) if log else img

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data, cmap=cmap)

    ax.set_title(f"{label}, slice {z}")
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    return fig, ax



def plot_array_slider(
    array,
    label: str = "Volume",
    log: bool = True,
    colorscale: str = "Viridis",
    width: int = 700,
    height: int = 700,
):
    """
    Plot a 3D numpy array with a Plotly layer slider.

    Parameters
    ----------
    array:
        3D array with shape (z, y, x).
    label:
        Plot title / label.
    log:
        If True, show log(1 + counts).
    colorscale:
        Plotly colorscale.
    """

    import numpy as np
    import plotly.graph_objects as go

    arr = np.asarray(array)

    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D array with shape (z, y, x), got {arr.shape}")

    arr_plot = np.log1p(arr) if log else arr
    z_count = arr_plot.shape[0]

    fig = go.Figure()

    for z in range(z_count):
        fig.add_trace(
            go.Heatmap(
                z=arr_plot[z],
                colorscale=colorscale,
                visible=(z == 0),
                colorbar=dict(title="log(1 + counts)" if log else "counts"),
            )
        )

    steps = []

    for z in range(z_count):
        steps.append(
            dict(
                method="update",
                args=[
                    {"visible": [i == z for i in range(z_count)]},
                    {"title": f"{label} — layer {z}"},
                ],
                label=str(z),
            )
        )

    fig.update_layout(
        title=f"{label} — layer 0",
        xaxis_title="X pixel",
        yaxis_title="Y pixel",
        yaxis=dict(scaleanchor="x", autorange="reversed"),
        width=width,
        height=height,
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Layer: "},
                pad={"t": 50},
                steps=steps,
            )
        ],
    )

    return fig


def plot_array_grid_slider(
    arrays: dict,
    log: bool = True,
    colorscale: str = "Viridis",
    ncols: int = 2,
    width: int = 900,
    height: int = 850,
):
    """
    Plot several 3D arrays in a grid with one shared layer slider.

    Parameters
    ----------
    arrays:
        Dict of {label: array}, each array shape must be (z, y, x).
    """

    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    labels = list(arrays.keys())

    if len(labels) == 0:
        raise ValueError("No arrays provided.")

    if len(labels) > 4:
        raise ValueError("This helper is intended for up to 4 arrays.")

    arrs = {label: np.asarray(arrays[label]) for label in labels}

    shapes = {label: arr.shape for label, arr in arrs.items()}

    if any(arr.ndim != 3 for arr in arrs.values()):
        raise ValueError(f"All arrays must be 3D with shape (z, y, x). Got: {shapes}")

    z_counts = {arr.shape[0] for arr in arrs.values()}

    if len(z_counts) != 1:
        raise ValueError(f"All arrays must have the same number of layers. Got: {shapes}")

    z_count = z_counts.pop()
    nrows = int(np.ceil(len(labels) / ncols))

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=labels,
        horizontal_spacing=0.05,
        vertical_spacing=0.08,
    )

    trace_indices_by_z = {z: [] for z in range(z_count)}

    for i, label in enumerate(labels):
        row = i // ncols + 1
        col = i % ncols + 1

        arr = arrs[label]
        arr_plot = np.log1p(arr) if log else arr

        for z in range(z_count):
            this_colorscale = (
                colorscale.get(label, "Viridis")
                if isinstance(colorscale, dict)
                else colorscale
            )
            trace = go.Heatmap(
                z=arr_plot[z],
                colorscale=this_colorscale,
                visible=(z == 0),
                showscale=(i == len(labels) - 1),
                colorbar=dict(title="log(1 + counts)" if log else "counts"),
            )

            fig.add_trace(trace, row=row, col=col)
            trace_indices_by_z[z].append(len(fig.data) - 1)

    steps = []

    for z in range(z_count):
        visible = [False] * len(fig.data)

        for idx in trace_indices_by_z[z]:
            visible[idx] = True

        steps.append(
            dict(
                method="update",
                args=[
                    {"visible": visible},
                    {"title": f"Layer {z}"},
                ],
                label=str(z),
            )
        )

    fig.update_layout(
        title="Layer 0",
        width=width,
        height=height,
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Layer: "},
                pad={"t": 50},
                steps=steps,
            )
        ],
    )

    fig.update_yaxes(autorange="reversed", scaleanchor="x")

    return fig

def plot_rgb_overlay(
    red,
    green,
    blue,
    red_label="Red",
    green_label="Green",
    blue_label="Blue",
    log: bool = True,
    normalize: bool = True,
    figsize=(8, 8),
):
    """
    Create an RGB overlay image from three 2D ion images.

    Parameters
    ----------
    red, green, blue:
        2D numpy arrays.
    """

    import numpy as np
    import matplotlib.pyplot as plt

    def prepare(arr):
        arr = np.asarray(arr, dtype=float)

        if log:
            arr = np.log1p(arr)

        if normalize:
            vmax = arr.max()

            if vmax > 0:
                arr = arr / vmax

        return arr

    r = prepare(red)
    g = prepare(green)
    b = prepare(blue)

    rgb = np.dstack([r, g, b])

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(rgb)
    ax.set_title(
        f"RGB overlay\n"
        f"R={red_label}, G={green_label}, B={blue_label}"
    )

    ax.set_xticks([])
    ax.set_yticks([])

    return fig, ax


def plot_rgb_overlay_slider(
    red,
    green,
    blue,
    red_label="Red",
    green_label="Green",
    blue_label="Blue",
    log: bool = True,
    normalize: bool = True,
    width: int = 700,
    height: int = 700,
):
    """
    RGB overlay from three 3D arrays with one layer slider.

    Arrays must have shape (z, y, x).
    """

    import numpy as np
    import plotly.graph_objects as go

    def prepare(arr):
        arr = np.asarray(arr, dtype=float)

        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array (z, y, x), got {arr.shape}")

        if log:
            arr = np.log1p(arr)

        if normalize:
            vmax = arr.max()
            if vmax > 0:
                arr = arr / vmax

        return np.clip(arr, 0, 1)

    r = prepare(red)
    g = prepare(green)
    b = prepare(blue)

    if not (r.shape == g.shape == b.shape):
        raise ValueError(
            f"RGB arrays must have same shape. Got R={r.shape}, G={g.shape}, B={b.shape}"
        )

    z_count = r.shape[0]

    fig = go.Figure()

    for z in range(z_count):
        rgb = np.dstack([r[z], g[z], b[z]])

        fig.add_trace(
            go.Image(
                z=(rgb * 255).astype(np.uint8),
                visible=(z == 0),
            )
        )

    steps = []

    for z in range(z_count):
        steps.append(
            dict(
                method="update",
                args=[
                    {"visible": [i == z for i in range(z_count)]},
                    {"title": f"RGB overlay — layer {z}"},
                ],
                label=str(z),
            )
        )

    fig.update_layout(
        title=f"RGB overlay — layer 0<br>R={red_label}, G={green_label}, B={blue_label}",
        width=width,
        height=height,
        xaxis_title="X pixel",
        yaxis_title="Y pixel",
        yaxis=dict(scaleanchor="x", autorange="reversed"),
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Layer: "},
                pad={"t": 50},
                steps=steps,
            )
        ],
    )

    return fig


def plot_element_overlay_slider(
    background,
    overlays: dict,
    log: bool = True,
    normalize: bool = True,
    alpha: float = 0.65,
    colors: dict | None = None,
    width: int = 750,
    height: int = 750,
):
    """
    Plot total counts as grayscale background with colored element overlays.

    Parameters
    ----------
    background:
        3D array, e.g. volume.get("Total"), shape (z, y, x).
    overlays:
        Dict of {label: 3D array}, e.g. {"Au": au_volume, "Pt": pt_volume}.
    """

    import numpy as np
    import plotly.graph_objects as go

    if colors is None:
        colors = {
            "Au": (1.0, 0.75, 0.0),   # gold/yellow
            "Pt": (0.0, 0.8, 1.0),    # cyan
            "W": (1.0, 0.0, 1.0),     # magenta
            "Cr": (0.0, 1.0, 0.0),    # green
        }

    def prepare(arr):
        arr = np.asarray(arr, dtype=float)

        if arr.ndim != 3:
            raise ValueError(f"Expected 3D array (z, y, x), got {arr.shape}")

        if log:
            arr = np.log1p(arr)

        if normalize:
            vmax = arr.max()
            if vmax > 0:
                arr = arr / vmax

        return np.clip(arr, 0, 1)

    bg = prepare(background)

    prepared_overlays = {
        label: prepare(arr)
        for label, arr in overlays.items()
    }

    for label, arr in prepared_overlays.items():
        if arr.shape != bg.shape:
            raise ValueError(
                f"Overlay '{label}' has shape {arr.shape}, "
                f"but background has shape {bg.shape}"
            )

    z_count = bg.shape[0]

    fig = go.Figure()

    for z in range(z_count):
        rgb = np.dstack([bg[z], bg[z], bg[z]])

        for label, arr in prepared_overlays.items():
            color = colors.get(label, (1.0, 0.0, 0.0))

            color_img = np.zeros_like(rgb)
            color_img[..., 0] = color[0] * arr[z]
            color_img[..., 1] = color[1] * arr[z]
            color_img[..., 2] = color[2] * arr[z]

            mask = arr[z][..., None]
            rgb = rgb * (1 - alpha * mask) + color_img * (alpha * mask)

        rgb = np.clip(rgb, 0, 1)

        fig.add_trace(
            go.Image(
                z=(rgb * 255).astype(np.uint8),
                visible=(z == 0),
            )
        )

    steps = []

    for z in range(z_count):
        steps.append(
            dict(
                method="update",
                args=[
                    {"visible": [i == z for i in range(z_count)]},
                    {"title": f"Element overlay — layer {z}"},
                ],
                label=str(z),
            )
        )

    overlay_text = ", ".join(overlays.keys())

    fig.update_layout(
        title=f"Element overlay — layer 0<br>Background=Total, overlays={overlay_text}",
        width=width,
        height=height,
        xaxis_title="X pixel",
        yaxis_title="Y pixel",
        yaxis=dict(scaleanchor="x", autorange="reversed"),
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "Layer: "},
                pad={"t": 50},
                steps=steps,
            )
        ],
    )

    return fig