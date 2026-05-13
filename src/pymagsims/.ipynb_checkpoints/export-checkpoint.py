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