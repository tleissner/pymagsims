# pymagsims

**pymagsims** is a Python toolkit for analysing MagSIMS (magnetic sector secondary ion mass spectrometry) data. It provides a unified workflow for:

* spectrum analysis and peak detection
* isotope / element identification
* raw event-based image reconstruction
* mass-filtered ion imaging
* interactive exploration and manual peak binning

The package is designed for **Jupyter-based exploratory analysis** as well as reproducible scientific workflows.

---

## Features

* 📈 Load and analyse MagSIMS spectra (main analysis files)
* 🔍 Peak detection and isotope assignment
* 🧪 Automatic bin generation from identified peaks
* 🖼️ Reconstruction of ion images from raw FPD event data
* 🎯 Mass-filtered and isotope-specific imaging
* 📊 Processed image import (FPD CSV export)
* ⚡ Interactive plotting (Plotly) and manual ROI binning
* 🧩 Modular, object-oriented design

---

## Installation

This project uses **uv** for environment and dependency management.

### 1. Clone the repository

```bash
git clone https://github.com/tleissner/pymagsims.git
cd pymagsims
```

### 2. Install dependencies

```bash
uv sync
```

### 3. (Optional) Install Jupyter

```bash
uv add jupyterlab ipykernel
uv run python -m jupyterlab
```

---

## Quickstart

```python
from pymagsims import Spectrum
from pymagsims.isotopes import load_builtin_isotopes

spec = Spectrum.from_main_analysis_file("data/example.csv")

spec.plot(log_y=True)

isotopes = load_builtin_isotopes()

fig, ax, assignments = spec.plot_with_peaks(
    isotope_table=isotopes,
    log_y=True,
)
```

---

## Example notebooks

See the `notebooks/` folder for fully documented workflows:

* spectrum analysis
* peak identification
* raw image reconstruction
* mass-filtered imaging
* interactive analysis
* full end-to-end pipeline

---

## Data handling

* Small test data → stored in Git
* Medium example data → stored using Git LFS
* Large experimental datasets → should be stored externally

---

## Project status

This is an **early-stage research tool** under active development.
APIs may change as the workflow matures.

---

## Development note

This project was initially **“vibe-coded”** — rapidly prototyped with the assistance of modern AI tools to explore ideas and workflows efficiently.

The focus is now on:

* improving robustness
* adding tests
* refining APIs
* documenting scientific assumptions

---

## License

GPL-3.0-or-later

