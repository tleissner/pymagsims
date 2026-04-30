from pathlib import Path
from pymagsims.spectrum import Spectrum

DATA = Path(__file__).parent / "data"

def test_read_main_analysis_file():
    spec = Spectrum.from_main_analysis_file(DATA / "FPD_01_2604281458290.csv")

    assert not spec.data.empty
    assert list(spec.data.columns) == ["Channel", "Mass", "Amplitude"]
    assert "file_information" in spec.metadata
    assert spec.roi_table is not None