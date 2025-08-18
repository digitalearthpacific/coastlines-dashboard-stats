import os
from pathlib import Path

from antimeridian import fix_polygon
from dep_tools.grids import grid
from dep_tools.utils import search_across_180
from exactextract import exact_extract
import geopandas as gpd
import numpy as np
import odc.stac
import pandas as pd
import pystac_client
from tqdm import tqdm
import xarray as xr

tqdm.pandas()  # turn something on

from config import COASTLINES_FILE, EEZ, EQUAL_AREA_CRS, OUTPUT_DIR
from regional_rates_of_change import add_rates_of_change_calculations


def main(coastlines_file: Path = COASTLINES_FILE):
    hotspots = gpd.read_file(
        coastlines_file, layer="hotspots_zoom_3", engine="pyogrio", use_arrow=True
    )
    contiguous_hotspots = calculate_contiguous_hotspots(hotspots)
    # At issue is how to deal with hotspots which cross grid boundaries.
    # A few approaches:
    # 1. Calculate twice and only take one value
    # 2. Only take the id of the grid cell with contains more of the hotspot (and
    #    assume when searching it will pull data for the neighboring cell)
    # 3. Split on the boundary and fix in rollup
    #
    # Going with approach #1 for now since it's the easiest to code (I think)

    ratesofchange = gpd.read_file(
        coastlines_file, layer="rates_of_change", engine="pyogrio", use_arrow=True
    )
    contiguous_hotspots = add_rates_of_change_calculations(
        contiguous_hotspots, ratesofchange
    )
    contiguous_hotspots = intersect_with_grid(contiguous_hotspots)

    # Process by each column row, to conserve loading time
    total_pop = contiguous_hotspots.groupby(
        ["column", "row"], group_keys=False
    ).progress_apply(total_population)
    # Duplicate indices here, but data are the same
    contiguous_hotspots["total_population"] = (
        total_pop.groupby(total_pop.index)
        .first()
        .reindex(contiguous_hotspots.index, fill_value=0)
    )

    contiguous_hotspots["building_counts"] = count_buildings(contiguous_hotspots)
    mangrove_area_ha = contiguous_hotspots.groupby(
        ["column", "row"], group_keys=False
    ).progress_apply(mangroves_area)
    contiguous_hotspots["mangrove_area_ha"] = (
        mangrove_area_ha.groupby(mangrove_area_ha.index)
        .first()
        .reindex(contiguous_hotspots.index, fill_value=0)
    )

    contiguous_hotspots = contiguous_hotspots.sjoin(EEZ[["geometry", "ISO_Ter1"]])

    contiguous_hotspots.to_file(OUTPUT_DIR / "output/contiguous_hotspots.gpkg")

    build_tiles(contiguous_hotspots, OUTPUT_DIR / "contiguous_hotspots.pmtiles")


def intersect_with_grid(non_point_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    grid_gdf = grid(
        intersect_with=non_point_gdf.reset_index()[["geometry"]],
        return_type="GeoDataFrame",
    ).drop_duplicates()

    return non_point_gdf.sjoin(
        grid_gdf.reset_index().rename(columns=dict(level_0="column", level_1="row"))
    ).drop("index_right", axis=1)


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
    contiguous_retreated_hotspots = (
        gpd.GeoDataFrame(
            geometry=[retreated_hotspots.geometry.union_all()],
            crs=hotspots.crs,
        ).explode(ignore_index=True)
        #   .assign(direction="retreat")
    )

    # Do the same for those which showed coastal growth
    grown_hotspots = buffered_hotspots[buffered_hotspots.rate_time > 0]
    contiguous_grown_hotspots = (
        gpd.GeoDataFrame(
            geometry=[grown_hotspots.geometry.union_all()],
            crs=hotspots.crs,
        ).explode(ignore_index=True)
        #   .assign(direction="growth")
    )

    # combine and return
    return pd.concat(
        [contiguous_retreated_hotspots, contiguous_grown_hotspots], ignore_index=True
    )


def count_buildings(gdf: gpd.GeoDataFrame) -> pd.Series:
    return (
        gdf.sjoin(buildings.to_crs(gdf.crs))
        .reset_index(names=["index"])
        .groupby("index")["index_right"]
        .count()
        .reindex(gdf.index, fill_value=0)
    )


def mangroves_area(gdf: gpd.GeoDataFrame):
    client = pystac_client.Client.open("https://stac.digitalearthpacific.org")
    items = search_across_180(
        gdf, client, collections=["dep_s2_mangroves"], datetime="2024"
    )
    if len(items) == 0:
        output = pd.DataFrame(np.zeros((len(gdf), 1)), columns=["sum"])
    else:
        mangroves = (
            odc.stac.load(
                items, crs=EQUAL_AREA_CRS, geopolygon=gdf.to_crs(EQUAL_AREA_CRS)
            ).squeeze(drop=True)
        ) > 0
        ha_per_sqm = 1 / 10_000
        cell_area_sqm = abs(np.prod(mangroves.odc.geobox.resolution.xy)).item()
        cell_area_ha = cell_area_sqm * ha_per_sqm
        mangroves_area_ha = mangroves * cell_area_ha
        output = exact_extract(
            mangroves_area_ha, gdf.to_crs(mangroves.odc.crs), ["sum"], output="pandas"
        )
        # exact_extract doesn't preserve index
    output.index = gdf.index
    return output


def total_population(gdf: gpd.GeoDataFrame):
    print(f"{gdf.column.iloc[0]},{gdf.row.iloc[0]}")
    client = pystac_client.Client.open("https://stac.staging.digitalearthpacific.io")
    items = search_across_180(gdf, client, collections=["dep_pdhhdx_population"])
    if len(items) == 0:
        return pd.DataFrame(np.zeros((1, 1)), columns=["sum"])
    pop_per_sqkm = (
        odc.stac.load(items, crs=EQUAL_AREA_CRS, geopolygon=gdf.to_crs(EQUAL_AREA_CRS))
        .max(dim="time")
        .squeeze(drop=True)
        .pop_per_sqkm
    )
    sqkm_per_sqm = 1 / 1_000_000
    cell_area_sqm = abs(np.prod(pop_per_sqkm.odc.geobox.resolution.xy)).item()
    cell_area_sqkm = cell_area_sqm * sqkm_per_sqm
    area_sqkm = xr.ones_like(pop_per_sqkm) * cell_area_sqkm
    pop_count = pop_per_sqkm * area_sqkm
    output = exact_extract(
        pop_count,
        gdf.to_crs(pop_count.odc.crs),
        ["sum"],
        output="pandas",
    )
    # exact_extract doesn't preserve index
    output.index = gdf.index
    return output


def build_tiles(stats: gpd.GeoDataFrame, output_file: Path) -> None:
    stats = stats.copy().to_crs(4326)
    stats["geometry"] = stats.geometry.apply(fix_polygon)
    output_geojson_path = output_file.parent / f"{output_file.stem}.geojson"
    output_pmtile_path = output_file.parent / f"{output_file.stem}.pmtiles"
    stats.to_file(output_geojson_path)
    os.system(f"tippecanoe {output_geojson_path} -pi -z13 -f -o {output_pmtile_path}")


if __name__ == "__main__":
    main()
