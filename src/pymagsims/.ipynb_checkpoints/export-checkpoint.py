# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile


def export_imagej_hyperstack(
    arrays: dict[str, np.ndarray],
    path: str | Path,
    dtype: str = "float32",
):
    from pathlib import Path
    import numpy as np
    import tifffile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(arrays.keys())

    # Input arrays are each (z, y, x)
    stack_czyx = np.stack(
        [np.asarray(arrays[label]) for label in labels],
        axis=0,
    )

    # Convert CZYX -> TZCYX with T=1
    stack_tzcyx = np.moveaxis(stack_czyx, 0, 1)  # Z, C, Y, X
    stack_tzcyx = stack_tzcyx[np.newaxis, ...]   # T, Z, C, Y, X

    stack_tzcyx = stack_tzcyx.astype(dtype)

    tifffile.imwrite(
        path,
        stack_tzcyx,
        imagej=True,
        metadata={
            "axes": "TZCYX",
            "channels": len(labels),
            "slices": stack_tzcyx.shape[1],
            "frames": 1,
            "Labels": labels,
        },
    )

    return path


def export_paraview_vti(
    arrays: dict,
    path,
    spacing=(1.0, 1.0, 1.0),
):
    """
    Export multiple 3D SIMS volumes to ParaView VTI format.

    Parameters
    ----------
    arrays:
        Dict of {label: 3D array} with shape (z, y, x)

    spacing:
        Physical voxel spacing (z, y, x)
    """

    from pathlib import Path

    import numpy as np
    import pyvista as pv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(arrays.keys())

    first = np.asarray(arrays[labels[0]])

    if first.ndim != 3:
        raise ValueError(
            f"Expected 3D arrays (z, y, x), got {first.shape}"
        )

    nz, ny, nx = first.shape

    grid = pv.ImageData()

    # VTK uses x,y,z ordering
    grid.dimensions = (nx, ny, nz)

    # voxel spacing
    grid.spacing = spacing[::-1]

    grid.origin = (0, 0, 0)

    for label, arr in arrays.items():
        arr = np.asarray(arr)

        if arr.shape != (nz, ny, nx):
            raise ValueError(
                f"Array '{label}' shape {arr.shape} "
                f"does not match {(nz, ny, nx)}"
            )

        # Convert z,y,x -> x,y,z flattening
        vtk_arr = np.transpose(arr, (2, 1, 0)).flatten(order="F")

        grid.point_data[label] = vtk_arr

    grid.save(path)

    return path