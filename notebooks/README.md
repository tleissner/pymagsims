# pymagsims example notebooks

These notebooks demonstrate the main software capabilities:

1. `01_quickstart.ipynb` — minimal first workflow
2. `02_spectrum_analysis.ipynb` — spectrum loading, metadata, calibration and plotting
3. `03_peak_identification.ipynb` — peak detection and isotope matching
4. `04_raw_image_processing.ipynb` — FPD raw event image loading
5. `05_mass_filtered_imaging.ipynb` — spectrum-calibrated ion images
6. `06_processed_image_import.ipynb` — processed FPD image CSV import
7. `07_interactive_analysis.ipynb` — Plotly and manual binning
8. `08_full_workflow.ipynb` — end-to-end workflow

Place example data in a project-level `data/` folder, for example:

```text
data/
├── FPD_01_2604281458290.csv
├── FPD_01_2604281458290.raw
├── FPD_image2.csv
└── FPD_image2.raw
```

Run Jupyter from the project root:

```bash
uv sync
uv run python -m jupyterlab
```