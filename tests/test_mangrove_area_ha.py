import geopandas as gpd
from pathlib import Path

from src.contiguous_hotspots import mangroves_area

TEST_DATA_DIRECTORY = Path(__file__).parent / "data"


def test_zero_area():
    zero_area = gpd.read_file(TEST_DATA_DIRECTORY / "mangrove_test_1.gpkg")
    area = mangroves_area(zero_area)
    assert area == 0
