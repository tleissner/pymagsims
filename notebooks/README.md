# pymagsims example notebooks

These notebooks demonstrate the main software capabilities and recommended workflows.

---

## Core workflows

1. `01_quickstart.ipynb`  
   Minimal first workflow and overview.

2. `02_spectrum_analysis.ipynb`  
   Spectrum loading, metadata, calibration and plotting.

3. `03_peak_identification.ipynb`  
   Peak detection and isotope matching.

4. `03_peak_identification_with_bin_export.ipynb`  
   Peak identification plus export of generated mass bins.

5. `04_raw_image_processing.ipynb`  
   FPD raw event image loading.

6. `05_mass_filtered_imaging.ipynb`  
   Spectrum-calibrated ion image generation.

7. `06_processed_image_import.ipynb`  
   Processed FPD image CSV import.

8. `07_interactive_analysis.ipynb`  
   Plotly-based interactive analysis and manual binning.

9. `07_interactive_analysis_with_manual_bin_export.ipynb`  
   Interactive manual binning with bin export.

10. `08_full_workflow.ipynb`  
    End-to-end workflow example.

---

## 3D SIMS workflows

11. `09_3d_raw_images_from_spectrum_bins_with_slider.ipynb`  
    Reconstruct 3D ion volumes from calibrated spectrum bins with interactive layer sliders.

12. `09_3d_raw_images_from_spectrum_bins_with_saved_bins.ipynb`  
    Reconstruct 3D ion volumes from previously saved bins.

13. `10_3d_raw_images_channel_workflow.ipynb`  
    Channel-based workflow without mass calibration.

14. `11_3d_raw_images_placeholder_calibration_peak_bins.ipynb`  
    Generate placeholder calibration, perform peak assignment and create bins.

15. `12_3d_workflow_from_csv_calibration.ipynb`  
    Complete workflow using channel–mass calibration extracted from matching 3D CSV export.

16. `13_3d_element_maps.ipynb`  
    Element-specific 3D visualization, layer sliders and grid plotting.

17. `14_3d_overlays.ipynb`  
    RGB overlays, grayscale total-count overlays and multi-element visualization.

18. `15_3d_export_imagej_paraview.ipynb`  
    Export reconstructed volumes to:

    - ImageJ/Fiji hyperstacks (`.tif`)
    - ParaView voxel volumes (`.vti`)

19. `16_3d_paraview_notes.ipynb`  
    Notes and recommended ParaView workflows for:

    - volume rendering
    - contour surfaces
    - clipping
    - interface visualization
    - slice inspection

---

## Example data layout

Place example data in a project-level `data/` folder, for example:

```text
data/
├── 3d/
│   ├── 202505077-MEMS-011_neg_500mT_3dimageImage_1.raw
│   ├── 202505077-MEMS-011_neg_500mT_3dimageImage_2.raw
│   └── ...
├── bins/
├── calibration/
├── export/
├── FPD_01_2604281458290.csv
├── FPD_01_2604281458290.raw
├── FPD_image2.csv
└── FPD_image2.raw