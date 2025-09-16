"""Derive contiguous hotspots and calculate statistics for each one."""

import os
from pathlib import Path

from antimeridian import fix_multi_polygon, fix_polygon
from dep_tools.grids import grid
from dep_tools.utils import search_across_180
from exactextract import exact_extract
import geopandas as gpd
import numpy as np
import odc.stac
import pandas as pd
import pystac_client
from s3fs import S3FileSystem
from shapely import voronoi_polygons
from shapely.geometry import MultiPoint
from tqdm import tqdm
import xarray as xr

tqdm.pandas()  # turn tqdm on for pandas ops

from common import remove_exclusions

from config import (
    BUILDINGS,
    CHANGE_THRESHOLD_KM_PER_YR,
    COASTLINES_FILE,
    EEZ,
    EQUAL_AREA_CRS,
    OUTPUT_DIR,
    S3_PATH,
)
from regional_rates_of_change import calculate_rates_of_change_over_polygons


def main(
    coastlines_file: Path = COASTLINES_FILE, hotspots_layer: str = "hotspots_zoom_3"
):
    hotspots = gpd.read_file(
        coastlines_file, layer=hotspots_layer, engine="pyogrio", use_arrow=True
    )
    hotspots = remove_exclusions(hotspots)

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

    contiguous_hotspots = calculate_rates_of_change_over_polygons(
        contiguous_hotspots, ratesofchange
    )

    # Drop distance columns to save space
    cols_to_drop = [
        col for col in contiguous_hotspots.columns if col.startswith("dist_")
    ]
    contiguous_hotspots = gpd.GeoDataFrame(
        contiguous_hotspots.drop(columns=cols_to_drop)
    )

    contiguous_hotspots = intersect_with_grid(contiguous_hotspots)

    # Get total population for each contiguous hotspot
    # Process by each column-row, to conserve loading time
    total_pop = contiguous_hotspots.groupby(
        ["column", "row"], group_keys=False
    ).progress_apply(total_population)
    # Duplicate indices here, but data are the same
    contiguous_hotspots["total_population"] = (
        total_pop.groupby(total_pop.index)
        .first()
        .reindex(contiguous_hotspots.index, fill_value=0)
    )

    # Get total mangrove area for each contiguous hotspot
    contiguous_hotspots["building_counts"] = count_buildings(contiguous_hotspots)
    mangrove_area_ha = contiguous_hotspots.groupby(
        ["column", "row"], group_keys=False
    ).progress_apply(mangroves_area)
    contiguous_hotspots["mangrove_area_ha"] = (
        mangrove_area_ha.groupby(mangrove_area_ha.index)
        .first()
        .reindex(contiguous_hotspots.index, fill_value=0)
    )

    # Add country code
    contiguous_hotspots = contiguous_hotspots.sjoin(EEZ[["geometry", "ISO_Ter1"]])

    # remove features duplicated across grid cells
    contiguous_hotspots = gpd.GeoDataFrame(
        contiguous_hotspots.groupby("uid").first(),
        geometry="geometry",
        crs=contiguous_hotspots.crs,
    )

    contiguous_hotspots_geopackage = OUTPUT_DIR / "contiguous_hotspots.gpkg"
    contiguous_hotspots.to_file(contiguous_hotspots_geopackage)

    contiguous_hotspots_pmtiles = OUTPUT_DIR / "contiguous_hotspots.pmtiles"
    build_tiles(contiguous_hotspots, contiguous_hotspots_pmtiles)

    s3 = S3FileSystem()
    s3.put(contiguous_hotspots_geopackage, S3_PATH)
    s3.put(contiguous_hotspots_pmtiles, S3_PATH)


def intersect_with_grid(non_point_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Intersect a GeoDataFrame with the DEP grid, adding "column" & "row" attributes.

    Args:
        non_point_gdf: A non-point GeoDataFrame.

    Returns:
        The input with columns "column" & "row" added, indicating which tile
        of the DEP grid the corresponding shape falls into. If a shape falls
        into multiple tiles, it is split by tile boundary.

    """
    grid_gdf = gpd.GeoDataFrame(
        grid(
            intersect_with=gpd.GeoDataFrame(non_point_gdf.reset_index()[["geometry"]]),
            return_type="GeoDataFrame",
        )
    ).drop_duplicates()

    return gpd.GeoDataFrame(
        non_point_gdf.sjoin(
            gpd.GeoDataFrame(
                grid_gdf.reset_index().rename(
                    columns=dict(level_0="column", level_1="row")
                )
            )
        ).drop("index_right", axis=1)
    )


def calculate_contiguous_hotspots(
    hotspots: gpd.GeoDataFrame,
    lower_change_threshold: float = CHANGE_THRESHOLD_KM_PER_YR,
) -> gpd.GeoDataFrame:
    """Create non-overlapping sets of like-directioned hotspots.

    Only input points with good certainty, significant relationships,
    and absolute change above the lower change threshold area kept.

    Args:
        hotspots: A hotspots GeoDataFrame, as would be created by
            dep_coastlines.continental. Should have point geometry and
            sig_time & certainty columns.
        lower_change_threshold: Rate of change values with absolute change
            lower than this are removed from processing.

    Returns:
        A GeoDataFrame where overlapping shapes are unioned, according to
        whether they represent growth or retreat.
    """
    # Only keep hotspots with significant change and good certainty
    # and change above threshold values
    good_hotspots = hotspots[
        (hotspots.sig_time < 0.01)
        & (hotspots.certainty == "good")
        & (abs(hotspots.rate_time) >= lower_change_threshold)
    ][["geometry", "rate_time"]].copy()

    #    radius = hotspots.radius_m.iloc[0]
    radius = 500
    # Buffer each by the original hotspot radius
    buffered_hotspots = good_hotspots.copy()
    buffered_hotspots["geometry"] = buffered_hotspots.geometry.buffer(
        radius,
    )

    # Prep to remove overlapping retreat & growth polygons by
    # creating voronoi polygons across all non-zero points and
    # splitting them into retreat and growth geometries.
    # With current code, threshold implies these are non-zero
    voronoi_nonzero_hotspots = good_hotspots[good_hotspots.rate_time != 0].copy()
    voronoi_nonzero_hotspots["geometry"] = voronoi_polygons(
        # fun fact: MultiPoint preserves order but .union_all() does not
        # https://github.com/shapely/shapely/issues/703
        MultiPoint(voronoi_nonzero_hotspots.geometry),
        ordered=True,
    ).geoms

    closer_to_growth = voronoi_nonzero_hotspots[
        voronoi_nonzero_hotspots.rate_time > 0
    ].geometry.union_all()

    closer_to_retreat = voronoi_nonzero_hotspots[
        voronoi_nonzero_hotspots.rate_time < 0
    ].geometry.union_all()

    # Select those which showed coastal retreat. Union and split into
    # non-touching polygons. Then clip by the appropriate voronoi polygon
    # sets.
    retreated_hotspots = buffered_hotspots[buffered_hotspots.rate_time < 0]
    contiguous_retreated_hotspots = (
        gpd.GeoDataFrame(
            geometry=[retreated_hotspots.geometry.union_all()],
            crs=hotspots.crs,
        )
        .explode(ignore_index=True)
        .clip(closer_to_retreat)
    )

    # Do the same for those which showed coastal growth
    grown_hotspots = buffered_hotspots[buffered_hotspots.rate_time > 0]
    contiguous_grown_hotspots = (
        gpd.GeoDataFrame(
            geometry=[grown_hotspots.geometry.union_all()],
            crs=hotspots.crs,
        )
        .explode(ignore_index=True)
        .clip(closer_to_growth)
    )

    # combine and return
    return gpd.GeoDataFrame(
        pd.concat(
            [contiguous_retreated_hotspots, contiguous_grown_hotspots],
            ignore_index=True,
        )
    )


def count_buildings(
    gdf: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame = BUILDINGS
) -> pd.Series:
    """Count the number of "buildings" in each shape of a GeoDataFrame.

    Args:
        gdf: A polygon GeoDataFrame.
        buildings: A GeoDataFrame where rows represent buildings.

    Returns:
        A series indexed the same as `gdf`, with the value representing
        the count of building features in the corresponding polygon.
    """
    return pd.Series(
        gdf.sjoin(buildings.to_crs(str(gdf.crs)))
        .reset_index(names=["index"])
        .groupby("index")["index_right"]
        .count()
        .reindex(gdf.index, fill_value=0)
    )


def mangroves_area(gdf: gpd.GeoDataFrame):
    """Calculate the area of mangroves in each shape of a GeoDataFrame.

    Areas are calculated using the DE Pacific Mangroves product for 2024. Areas
    with codes 1 & 2 (open & closed mangroves) are considered mangroves. As such,
    area estimates may be less than Global Mangrove Watch.

    Args:
        gdf: A polygon/multipolygon GeoDataFrame

    Returns:

    """
    client = pystac_client.Client.open("https://stac.digitalearthpacific.org")
    items = search_across_180(
        gdf, client, collections=["dep_s2_mangroves"], datetime="2024"
    )
    if len(items) == 0:
        output = pd.DataFrame(np.zeros((len(gdf), 1)), columns=["sum"])
    else:
        dep_s2_mangrove_codes = [1, 2]
        mangroves = (
            odc.stac.load(
                items, crs=EQUAL_AREA_CRS, geopolygon=gdf.to_crs(EQUAL_AREA_CRS)
            )
            .squeeze(drop=True)
            .isin(dep_s2_mangrove_codes)
        )
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
    """Use tippecanoe to build pmtiles (and geojson) for the output stats.

    Input geometry is fixed using :py:func:`antimeridian.fix_polygon` before
    assembling tiles.

    Args:
        stats: Any GeoDataFrame
        output_file: The path of the output file.
    """
    stats = stats.copy().to_crs(4326)
    stats["geometry"] = stats.geometry.apply(
        lambda geom: (
            fix_polygon(geom)
            if geom.geom_type == "Polygon"
            else fix_multi_polygon(geom)
        )
    )
    output_geojson_path = output_file.parent / f"{output_file.stem}.geojson"
    output_pmtile_path = output_file.parent / f"{output_file.stem}.pmtiles"
    stats.to_file(output_geojson_path)
    os.system(f"tippecanoe {output_geojson_path} -pi -z13 -f -o {output_pmtile_path}")


if __name__ == "__main__":
    main()
