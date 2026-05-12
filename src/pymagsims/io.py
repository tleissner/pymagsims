# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from .image import SIMSImage
from .spectrum import Spectrum


def read_main_analysis_file(
    path: str | Path,
    encoding: str = "latin1",
) -> Spectrum:
    path = Path(path)

    with path.open("r", encoding=encoding) as f:
        lines = f.readlines()

    end_idx = None
    spectrum_header_idx = None

    for i, line in enumerate(lines):
        if line.strip().startswith("%end"):
            end_idx = i

        if line.strip().startswith("Channel;Mass;Amplitude"):
            spectrum_header_idx = i
            break

    if spectrum_header_idx is None:
        raise ValueError("Could not find spectrum header: 'Channel;Mass;Amplitude'")

    metadata = _parse_main_metadata(lines[:7])
    roi_table = _parse_roi_block(lines[7:end_idx]) if end_idx is not None else None

    data = pd.read_csv(
        path,
        sep=";",
        skiprows=spectrum_header_idx,
        encoding=encoding,
    )

    data = data.loc[:, ["Channel", "Mass", "Amplitude"]]
    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    data["Channel"] = data["Channel"].astype(int)

    return Spectrum(
        data=data,
        metadata=metadata,
        roi_table=roi_table,
        name=path.stem,
    )


def read_raw_file(
    path: str | Path,
    calibration: Spectrum | pd.DataFrame | None = None,
    min_channel: int | None = None,
    max_channel: int | None = None,
    encoding: str = "latin1",
) -> Spectrum:
    path = Path(path)

    raw_events = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["Channel", "PixelX", "PixelY"],
        encoding=encoding,
    )

    raw_events = raw_events.apply(pd.to_numeric, errors="coerce").dropna()
    raw_events = raw_events.astype(
        {
            "Channel": int,
            "PixelX": int,
            "PixelY": int,
        }
    )

    spectrum_data = spectrum_from_raw_events(
        raw_events,
        calibration=calibration,
        min_channel=min_channel,
        max_channel=max_channel,
    )

    metadata = {
        "source": "raw",
        "n_events": int(len(raw_events)),
        "channel_min": int(raw_events["Channel"].min()),
        "channel_max": int(raw_events["Channel"].max()),
        "pixel_x_min": int(raw_events["PixelX"].min()),
        "pixel_x_max": int(raw_events["PixelX"].max()),
        "pixel_y_min": int(raw_events["PixelY"].min()),
        "pixel_y_max": int(raw_events["PixelY"].max()),
    }

    return Spectrum(
        data=spectrum_data,
        metadata=metadata,
        raw_events=raw_events,
        name=path.stem,
    )


def spectrum_from_raw_events(
    raw_events: pd.DataFrame,
    calibration: Spectrum | pd.DataFrame | None = None,
    min_channel: int | None = None,
    max_channel: int | None = None,
) -> pd.DataFrame:
    counts = raw_events["Channel"].value_counts().sort_index()

    if min_channel is None:
        min_channel = int(counts.index.min())

    if max_channel is None:
        max_channel = int(counts.index.max())

    channels = np.arange(min_channel, max_channel + 1)

    data = pd.DataFrame({"Channel": channels})
    data["Amplitude"] = data["Channel"].map(counts).fillna(0).astype(int)

    if calibration is not None:
        calibration_df = calibration.data if isinstance(calibration, Spectrum) else calibration
        calibration_df = calibration_df[["Channel", "Mass"]].copy()
        calibration_df["Channel"] = calibration_df["Channel"].astype(int)

        data = data.merge(calibration_df, on="Channel", how="left")
    else:
        data["Mass"] = np.nan

    return data[["Channel", "Mass", "Amplitude"]]


def _parse_main_metadata(lines: list[str]) -> dict:
    return {
        "file_information": _parse_key_value_line(lines[0]),
        "analysis_parameters": _parse_header_value_pair(lines[1], lines[2]),
        "sims_metrics": _parse_header_value_pair(lines[3], lines[4]),
        "platform_metrics": _parse_header_value_pair(lines[5], lines[6]),
    }


def _parse_key_value_line(line: str) -> dict:
    parts = _clean_parts(line)
    result = {}

    for i in range(0, len(parts), 2):
        key = parts[i]
        result[key] = _convert_value(parts[i + 1]) if i + 1 < len(parts) else None

    return result


def _parse_header_value_pair(header_line: str, value_line: str) -> dict:
    headers = _clean_parts(header_line)
    values = _clean_parts(value_line)

    return {
        key: _convert_value(value)
        for key, value in zip(headers, values)
    }


def _parse_roi_block(lines: list[str]) -> pd.DataFrame:
    entries = []
    current = {}

    roi_keys = {
        "Color",
        "Name",
        "Mass",
        "CH min",
        "CH max",
        "Mass min",
        "Mass max",
        "Integration",
        "Max",
        "FWHM",
    }

    for line in lines:
        parts = _clean_parts(line)

        if len(parts) < 2:
            continue

        key, value = parts[0], parts[1]

        if key == "Color" and current:
            entries.append(current)
            current = {}

        if key in roi_keys:
            current[key] = _convert_value(value)

    if current:
        entries.append(current)

    return pd.DataFrame(entries)


def _clean_parts(line: str) -> list[str]:
    return [
        part.strip()
        for part in line.strip().split(";")
        if part.strip() != ""
    ]


def _convert_value(value: str):
    try:
        number = float(value)

        if number.is_integer():
            return int(number)

        return number

    except ValueError:
        return value
    

def read_fpd_image_csv(
    path: str | Path,
    encoding: str = "latin1",
) -> SIMSImage:
    """
    Read processed FPD image CSV export.

    Expected structure:
        rows 1-7      metadata
        rows 8-17     ROI table
        row 18        %end
        rows 19-21    spectrum: Channel, Mass, Count
        following     image blocks:
                        label row
                        image rows
    """

    path = Path(path)

    with path.open("r", encoding=encoding) as f:
        lines = f.readlines()

    metadata = _parse_main_metadata(lines[:7])
    roi_table = _parse_roi_block(lines[7:17])

    spectrum = _parse_fpd_image_spectrum(lines)

    size = int(metadata["analysis_parameters"]["Size"])

    image_start_idx = _find_image_start_index(lines)
    images = _parse_fpd_image_blocks(
        lines=lines[image_start_idx:],
        size=size,
    )

    sims_image = SIMSImage(
        images=images,
        metadata=metadata,
        roi_table=roi_table,
        name=path.stem,
    )
    sims_image.spectrum = spectrum

    return sims_image


def _parse_fpd_image_spectrum(lines: list[str]) -> Spectrum:
    channel_line_idx = None

    for i, line in enumerate(lines):
        if line.startswith("Channel;"):
            channel_line_idx = i
            break

    if channel_line_idx is None:
        raise ValueError("Could not find spectrum block starting with 'Channel;'")

    channel_values = _clean_parts(lines[channel_line_idx])[1:]
    mass_values = _clean_parts(lines[channel_line_idx + 1])[1:]
    count_values = _clean_parts(lines[channel_line_idx + 2])[1:]

    data = pd.DataFrame(
        {
            "Channel": channel_values,
            "Mass": mass_values,
            "Amplitude": count_values,
        }
    )

    data = data.apply(pd.to_numeric, errors="coerce").dropna()
    data["Channel"] = data["Channel"].astype(int)

    return Spectrum(
        data=data,
        metadata={},
        name="embedded_spectrum",
    )


def _find_image_start_index(lines: list[str]) -> int:
    """
    The image section starts after the Channel/Mass/Count spectrum block.
    """

    for i, line in enumerate(lines):
        if line.startswith("Channel;"):
            return i + 3

    raise ValueError("Could not locate image start after spectrum block.")


def _parse_fpd_image_blocks(
    lines: list[str],
    size: int,
) -> dict[str, np.ndarray]:
    """
    Parse image blocks.

    Each image block:
        Label
        size rows of image data

    Handles one or multiple ROI images.
    Ignores trailing labels without complete image data.
    """

    images = {}
    i = 0

    while i < len(lines):
        parts = _clean_parts(lines[i])

        if len(parts) != 1:
            i += 1
            continue

        label = str(parts[0])

        candidate_rows = lines[i + 1 : i + 1 + size]

        if len(candidate_rows) < size:
            break

        rows = []

        valid_block = True

        for row_line in candidate_rows:
            row_parts = _clean_parts(row_line)

            if len(row_parts) != size:
                valid_block = False
                break

            rows.append([float(v) for v in row_parts])

        if not valid_block:
            i += 1
            continue

        images[label] = np.array(rows, dtype=float)

        i += 1 + size

    return images


def read_fpd_depth_profile_csv(
    path,
    encoding: str = "latin1",
    n_channels: int = 12000,
):
    from pathlib import Path
    import pandas as pd

    from .spectrum import Spectrum
    from .depth_profile import SIMSDepthProfile

    path = Path(path)

    with path.open("r", encoding=encoding) as f:
        lines = f.readlines()

    metadata = _parse_main_metadata(lines[:7])
    roi_table = _parse_roi_block(lines[7:17])

    end_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("%end"):
            end_idx = i
            break

    if end_idx is None:
        raise ValueError("Could not find %end marker.")

    spectrum_start = end_idx + 1
    spectrum_end = spectrum_start + n_channels

    spectrum_rows = []

    for line in lines[spectrum_start:spectrum_end]:
        parts = _clean_parts(line)

        if len(parts) >= 3:
            spectrum_rows.append(parts[:3])

    spectrum_df = pd.DataFrame(
        spectrum_rows,
        columns=["Channel", "Mass", "Amplitude"],
    )

    spectrum_df = spectrum_df.apply(pd.to_numeric, errors="coerce").dropna()
    spectrum_df["Channel"] = spectrum_df["Channel"].astype(int)

    last_spectrum = Spectrum(
        data=spectrum_df,
        metadata=metadata.copy(),
        roi_table=roi_table,
        name=f"{path.stem}_last_spectrum",
    )

    profile_df = pd.read_csv(
        path,
        sep=";",
        skiprows=spectrum_end,
        encoding=encoding,
    )

    profile_df = profile_df.dropna(axis=1, how="all")
    profile_df = profile_df.loc[
        :, ~profile_df.columns.astype(str).str.contains("^Unnamed")
    ]

    profile_df = profile_df.apply(lambda col: pd.to_numeric(col, errors="coerce"))
    profile_df = profile_df.dropna(axis=0, how="all")

    return SIMSDepthProfile(
        profiles=profile_df,
        metadata=metadata,
        roi_table=roi_table,
        last_spectrum=last_spectrum,
        name=path.stem,
    )

def read_fpd_depth_profile_raw(
    path,
    roi_table=None,
    bins=None,
    spectrum=None,
    skip_initial: int = 0,
    encoding: str = "latin1",
):
    from pathlib import Path
    import numpy as np
    import pandas as pd

    from .depth_profile import SIMSDepthProfile
    from .raw_image import mass_to_channel

    path = Path(path)

    hist = pd.read_csv(
        path,
        sep=";",
        header=None,
        encoding=encoding,
    )

    hist = hist.dropna(axis=1, how="all")
    hist = hist.apply(pd.to_numeric, errors="coerce").fillna(0)

    if skip_initial:
        hist_used = hist.iloc[skip_initial:].reset_index(drop=True)
    else:
        hist_used = hist.reset_index(drop=True)

    profiles = pd.DataFrame(
        {
            "Acquisition point": np.arange(1, len(hist_used) + 1),
        }
    )

    source = bins if bins is not None else roi_table

    if source is None:
        profiles["Total"] = hist_used.sum(axis=1)
    else:
        for _, row in source.iterrows():
            label = str(row.get("Name", row.get("label", "ROI")))

            if "CH min" in row and "CH max" in row:
                ch_min = int(row["CH min"])
                ch_max = int(row["CH max"])
            elif "ch_min" in row and "ch_max" in row:
                ch_min = int(row["ch_min"])
                ch_max = int(row["ch_max"])
            elif "mass_min" in row and "mass_max" in row:
                if spectrum is None:
                    raise ValueError("spectrum is required for mass-based bins.")
                ch_min = mass_to_channel(spectrum, float(row["mass_min"]))
                ch_max = mass_to_channel(spectrum, float(row["mass_max"]))
            else:
                raise ValueError(
                    "Bins/ROI table must contain CH min/CH max, "
                    "ch_min/ch_max, or mass_min/mass_max."
                )

            if ch_max < ch_min:
                ch_min, ch_max = ch_max, ch_min

            col_min = max(ch_min - 1, 0)
            col_max = min(ch_max - 1, hist_used.shape[1] - 1)

            profiles[label] = hist_used.iloc[:, col_min:col_max + 1].sum(axis=1)

    metadata = {
        "source": "FPD depth profile raw histogram",
        "n_histograms_total": int(hist.shape[0]),
        "n_histograms_used": int(hist_used.shape[0]),
        "n_channels": int(hist.shape[1]),
        "skip_initial": int(skip_initial),
    }

    return SIMSDepthProfile(
        profiles=profiles,
        metadata=metadata,
        roi_table=roi_table,
        raw_histograms=hist_used,
        name=path.stem,
    )