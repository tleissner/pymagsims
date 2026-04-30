from pathlib import Path
from pymagsims import spectrum
from pymagsims.isotopes import load_builtin_isotopes

DATA = Path(__file__).parent / "data"

def test_peak_detection_and_assignment():
    spec = spectrum.Spectrum.from_main_analysis_file(DATA / "FPD_01_2604281458290.csv")
    isotopes = load_builtin_isotopes()

    peaks = spec.find_peaks(prominence=100, distance=5)
    assignments = spec.assign_peaks(isotopes, tolerance=0.3, prominence=100)

    assert not peaks.empty
    assert "measured_mass" in peaks.columns
    assert "isotope" in assignments.columns