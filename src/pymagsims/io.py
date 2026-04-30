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