from pathlib import Path
from pymagsims.spectrum import Spectrum
from pymagsims.isotopes import load_builtin_isotopes

DATA = Path(__file__).parent / "data"

def test_create_bins_from_assignments():
    spec = Spectrum.from_main_analysis_file(DATA / "FPD_01_2604281458290.csv")
    isotopes = load_builtin_isotopes()

    assignments = spec.assign_peaks(isotopes, tolerance=0.3, prominence=100)
    bins = spec.create_bins_from_assignments(assignments, width=0.3)

    assert "mass_min" in bins.columns
    assert "mass_max" in bins.columns