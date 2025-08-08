import os
from pathlib import Path
import urllib.request

from dep_tools.grids import grid
from dep_tools.utils import search_across_180
from exactextract import exact_extract
import geopandas as gpd
import numpy as np
import odc.stac
import pandas as pd
from pyogrio import read_dataframe
import pystac_client
import rioxarray
from tqdm import tqdm
import xarray as xr

tqdm.pandas()  # turn something on

from regional_rates_of_change import calculate_rates_of_change_over_polygons

os.makedirs("data", exist_ok=True)
CRS = 3832

coastlines_file = Path("data/dep_ls_coastlines_0-7-0-55.gpkg")
if not coastlines_file.exists():
    remote_coastlines_file = "https://s3.us-west-2.amazonaws.com/dep-public-data/dep_ls_coastlines/dep_ls_coastlines_0-7-0-55.gpkg"
    urllib.request.urlretrieve(remote_coastlines_file, coastlines_file)

eez_file = Path("data/country_boundary_eez.geojson")
if not eez_file.exists():
    read_dataframe(
        "https://pacificdata.org/data/dataset/964dbebf-2f42-414e-bf99-dd7125eedb16/resource/dad3f7b2-a8aa-4584-8bca-a77e16a391fe/download/country_boundary_eez.geojson"
    ).to_file(eez_file)
eez = gpd.read_file(eez_file).to_crs(CRS)

buildings_file = Path("data/dep_buildings_0-1-0.gpkg")
if not buildings_file.exists():
    remote_buildings_file = "https://dep-public-staging.s3.us-west-2.amazonaws.com/dep_osm_buildings/dep_buildings_0-1-0.gpkg"
    urllib.request.urlretrieve(remote_buildings_file, buildings_file)
buildings = gpd.read_file(buildings_file).to_crs(CRS)


def main():
    hotspots = gpd.read_file(
        coastlines_file, layer="hotspots_zoom_3", engine="pyogrio", use_arrow=True
    )
    contiguous_hotspots = calculate_contiguous_hotspots(hotspots)
    contiguous_hotspots = add_grid_column_and_row(contiguous_hotspots)
    # contiguous_hotspots = calculate_rates_of_change_over_polygons(contiguous_hotspots)
    # contiguous_hotspots["building_counts"] = count_buildings(contiguous_hotspots)
    # contiguous_hotspots["total_population"] = contiguous_hotspots.groupby(
    x = contiguous_hotspots.groupby(["column", "row"]).progress_apply(total_population)
    breakpoint()

    # bounds = eez.geometry.to_crs(4326).bounds


def add_grid_column_and_row(non_point_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    grid_gdf = grid(intersect_with=non_point_gdf, return_type="GeoDataFrame")
    return gpd.sjoin(
        non_point_gdf,
        grid_gdf.reset_index().rename(columns=dict(level_0="column", level_1="row")),
    )


def calculate_contiguous_hotspots(hotspots: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Only keep hotspots with significant change and good certainty
    buffered_hotspots = hotspots[
        (hotspots.sig_time < 0.01) & (hotspots.certainty == "good")
    ][["geometry", "rate_time"]].copy()

    radius = hotspots.radius_m.iloc[0]
    # Buffer each by the original hotspot radius
    buffered_hotspots["geometry"] = buffered_hotspots.geometry.buffer(radius)

    # Select those which showed coastal retreat, union, and split into
    # non-touching polygons
    retreated_hotspots = buffered_hotspots[buffered_hotspots.rate_time < 0]
    contiguous_retreated_hotspots = gpd.GeoDataFrame(
        geometry=[retreated_hotspots.geometry.union_all()],
        crs=hotspots.crs,
    ).explode(ignore_index=True)

    # Do the same for those which showed coastal growth
    grown_hotspots = buffered_hotspots[buffered_hotspots.rate_time > 0]
    contiguous_grown_hotspots = gpd.GeoDataFrame(
        geometry=[grown_hotspots.geometry.union_all()],
        crs=hotspots.crs,
    ).explode(ignore_index=True)

    # combine and return
    return pd.concat([contiguous_retreated_hotspots, contiguous_grown_hotspots])


def count_buildings(gdf: gpd.GeoDataFrame) -> pd.Series:
    return (
        gpd.sjoin(gdf.reset_index(), buildings, predicate="intersects", how="left")
        .groupby("index")["index_right"]
        .count()
    )


def mangroves_area(gdf: gpd.GeoDataFrame):
    client = pystac_client.Client.open("https://stac.digitalearthpacific.org")
    items = search_across_180(gdf, client, collections=["dep_s2_mangroves"])
    mangroves = (
        odc.stac.load(items, crs=8859, geopolygon=gdf.to_crs(8859)).squeeze(drop=True)
    ) > 0
    ha_per_sqm = 1 / 10_000
    cell_area_sqm = abs(np.prod(mangroves.odc.geobox.resolution.xy)).item()
    cell_area_ha = cell_area_sqm * ha_per_sqm
    mangroves_area_ha = cell_area_ha.where(mangroves)
    return exact_extract(
        mangroves_area_ha, gdf.to_crs(mangroves.odc.crs), ["sum"], output="pandas"
    )


def total_population(gdf: gpd.GeoDataFrame):
    try:
        client = pystac_client.Client.open(
            "https://stac.staging.digitalearthpacific.io"
        )
        items = search_across_180(gdf, client, collections=["dep_pdhhdx_population"])
        pop_per_sqkm = (
            odc.stac.load(items, crs=8859, geopolygon=gdf.to_crs(8859))
            .squeeze(drop=True)
            .pop_per_sqkm
        )
        sqkm_per_sqm = 1 / 1_000_000
        cell_area_sqm = abs(np.prod(pop_per_sqkm.odc.geobox.resolution.xy)).item()
        cell_area_sqkm = cell_area_sqm * sqkm_per_sqm
        area_sqkm = xr.ones_like(pop_per_sqkm) * cell_area_sqkm
        pop_count = pop_per_sqkm * area_sqkm
        return exact_extract(
            pop_count, gdf.to_crs(pop_count.odc.crs), ["sum"], output="pandas"
        )
    except:
        breakpoint()


if __name__ == "__main__":
    main()
