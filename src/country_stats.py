"""Calculate country-level statistics."""

import json
import math
from pathlib import Path

from antimeridian import bbox, fix_multi_polygon, fix_polygon
import geopandas as gpd
import pandas as pd
from s3fs import S3FileSystem

from common import (
    get_change_magnitude_summary,
    get_change_type_summary,
    make_outliers_nan,
)
from config import COASTLINES_FILE, EEZ, OUTPUT_DIR, __version__, S3_PATH


def country_level_stats(
    coastlines_file: Path = COASTLINES_FILE, eez: gpd.GeoDataFrame = EEZ
):
    """Derive country level statistics.

    Population, buildings, and mangrove area are calculated by summing across
    values within all contiguous hotspots within the country. Country-level
    distances are calculated across all

    Args:
        coastlines_file: The path to the coastlines data
        eez: Economic exclusion zones, used to define country boundaries.
    """
    # Sum population, buildings, and mangrove area in all contiguous hotspots
    # in the entire country
    hotspot_stats = (
        gpd.read_file(OUTPUT_DIR / "contiguous_hotspots.gpkg")
        .groupby("ISO_Ter1")[
            ["total_population", "building_counts", "mangrove_area_ha"]
        ]
        .sum()
        .round(2)
    )

    # Calculate median distances for all ratesofchange points in the country
    ratesofchange = gpd.read_file(
        coastlines_file, layer="rates_of_change", engine="pyogrio", use_arrow=True
    ).rename(dict(eez_territory="ISO_Ter1"), axis=1)
    roc_stats = summarise_roc(ratesofchange.groupby("ISO_Ter1"))

    # Estimate bounding box based on
    eez = eez.dissolve(by="ISO_Ter1")

    def fix_and_bbox(geometry):
        fixer = fix_polygon if geometry.geom_type == "Polygon" else fix_multi_polygon
        return bbox(fixer(geometry), force_over_antimeridian=True)

    eez["bbox"] = eez.geometry.to_crs(4326).apply(fix_and_bbox)

    output = hotspot_stats.join(roc_stats).join(eez[["bbox"]]).reset_index(names="id")
    output.to_csv(OUTPUT_DIR / "country_summaries.csv", index=False)
    _write_geojson(output, OUTPUT_DIR / "country_summaries.geojson")


def _write_geojson(df: pd.DataFrame, output_path: Path) -> None:
    """Write country-level geojson according to Matthew's specs.

    Args:
        df: Country-level stats
        output_path: Where to write the output. Output is also copied to S3.
    """

    def _convert_nan_to_null(value):
        return None if math.isnan(value) else value

    features = []
    for _, row in df.iterrows():
        feature = {
            "type": "Feature",
            "bbox": row["bbox"],
            "geometry": {"type": "Point", "coordinates": None},
            "properties": {
                "id": row["id"],
                "shoreline_change_direction": {
                    "percent_retreat": row["percent_retreat"],
                    "percent_retreat_non_sig": row["percent_retreat_non_sig"],
                    "percent_growth": row["percent_growth"],
                    "percent_growth_non_sig": row["percent_growth_non_sig"],
                    "percent_stable": row["percent_stable"],
                },
                "shoreline_change_magnitude": {
                    "high_change_km": row["high_change_km"],
                    "medium_change_km": row["medium_change_km"],
                    "low_change_km": row["low_change_km"],
                },
                "population_in_hotspots": row["total_population"],
                "number_of_buildings_in_hotspots": row["building_counts"],
                "mangrove_area_ha_in_hotspots": row["mangrove_area_ha"],
                "median_distances": {
                    str(year): _convert_nan_to_null(row[f"dist_{year}"])
                    for year in range(1999, 2024)
                },
            },
        }
        features.append(feature)

    geojson_obj = {"type": "FeatureCollection", "features": features}

    with open(output_path, "w") as f:
        json.dump(geojson_obj, f, indent=2)

    s3 = S3FileSystem()
    s3.put(output_path, S3_PATH)


def summarise_roc(roc: pd.api.typing.DataFrameGroupBy) -> pd.DataFrame:
    """Create summaries for each group of rates of change points.

    Args:
        roc: A grouped rates of change dataframe.

    Returns: A dataframe with the following columns:
        - dist_{year} : Median distances for all rates of change points and years
            where certainty is good. Outlier years are removed from each point.
        - percent_{growth|retreat|stable|growth_non_sig|retreat_non_sig} : The percent
            of points in the corresponding category. Significance is determined
            by the sig_time column.
        - {high|medium|low|non_sig}_km : The kilometers of coastline in each
            category.
    """

    def get_distance_stats(d: pd.DataFrame, fewest_values: int = 250) -> pd.Series:
        """Calculate median distances for each dist_{year} column in d, across all
        rows that have certainty "good".
        """
        filtered_d = d.apply(make_outliers_nan, axis="columns").loc[
            d.certainty == "good",
            d.columns.str.contains("dist_"),
        ]
        counts = filtered_d.count()
        median = filtered_d.median().round(2)
        median[counts < fewest_values] = float("nan")
        return median

    distance_stats = roc.apply(get_distance_stats)

    change_type_percentages = roc.apply(
        get_change_type_summary, include_groups=False
    ).unstack(fill_value=0)
    change_magnitude_percentages = roc.apply(
        get_change_magnitude_summary, include_groups=False
    ).unstack(fill_value=0)

    return distance_stats.join(change_type_percentages).join(
        change_magnitude_percentages
    )


if __name__ == "__main__":
    country_level_stats()
