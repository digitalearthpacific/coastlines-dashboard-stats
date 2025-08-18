import os
from pathlib import Path
import urllib.request

import geopandas as gpd
from pyogrio import read_dataframe

EQUAL_AREA_CRS = 8859
VERSION = "0.1.1"
OUTPUT_DIR = Path("data/output") / VERSION
os.makedirs(OUTPUT_DIR, exist_ok=True)

COASTLINES_FILE = Path("data/dep_ls_coastlines_0-7-0-55.gpkg")
if not COASTLINES_FILE.exists():
    remote_coastlines_file = "https://s3.us-west-2.amazonaws.com/dep-public-data/dep_ls_coastlines/dep_ls_coastlines_0-7-0-55.gpkg"
    urllib.request.urlretrieve(remote_coastlines_file, COASTLINES_FILE)

eez_file = Path("data/country_boundary_eez.geojson")
if not eez_file.exists():
    read_dataframe(
        "https://pacificdata.org/data/dataset/964dbebf-2f42-414e-bf99-dd7125eedb16/resource/dad3f7b2-a8aa-4584-8bca-a77e16a391fe/download/country_boundary_eez.geojson"
    ).to_file(eez_file)
EEZ = gpd.read_file(eez_file).to_crs(3832)

buildings_file = Path("data/dep_buildings_0-1-0.gpkg")
if not buildings_file.exists():
    remote_buildings_file = "https://dep-public-staging.s3.us-west-2.amazonaws.com/dep_osm_buildings/dep_buildings_0-1-0.gpkg"
    urllib.request.urlretrieve(remote_buildings_file, buildings_file)
BUILDINGS = gpd.read_file(buildings_file).to_crs(EQUAL_AREA_CRS)
