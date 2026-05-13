# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import pandas as pd
import periodictable as pt


def load_builtin_isotopes(max_atomic_number: int = 79) -> pd.DataFrame:
    """
    Load isotope table up to a given atomic number.

    Default max_atomic_number=79 includes elements up to gold (Au).

    Returns columns:
        atomic_number
        element
        isotope
        mass_number
        exact_mass
        abundance
    """

    rows = []

    for element in pt.elements:
        if element.number is None:
            continue

        if element.number > max_atomic_number:
            continue

        for isotope in element:
            if isotope.mass is None:
                continue

            abundance = isotope.abundance
            if abundance is None:
                abundance = 0.0

            rows.append(
                {
                    "atomic_number": int(element.number),
                    "element": element.symbol,
                    "isotope": f"{isotope.isotope}{element.symbol}",
                    "mass_number": int(isotope.isotope),
                    "exact_mass": float(isotope.mass),
                    "abundance": float(abundance),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["atomic_number", "mass_number"]
    ).reset_index(drop=True)

def filter_isotopes(
    isotopes,
    min_abundance: float = 0.5,
    max_atomic_number: int | None = None,
):
    filtered = isotopes.copy()

    filtered = filtered[
        filtered["abundance"].fillna(0) >= min_abundance
    ]

    if max_atomic_number is not None:
        filtered = filtered[
            filtered["atomic_number"] <= max_atomic_number
        ]

    return filtered.reset_index(drop=True)