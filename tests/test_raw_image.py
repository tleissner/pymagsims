from pathlib import Path
from pymagsims.raw_image import SIMSRawImage

DATA = Path(__file__).parent / "data"

def test_raw_image_reading():
    raw = SIMSRawImage.from_fpd_raw(DATA / "FPD_image2.raw")

    assert not raw.events.empty
    assert {"X", "Y", "Channel"}.issubset(raw.events.columns)
    assert raw.total_ion_image().sum() == len(raw.events)