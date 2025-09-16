"""Calculate country-level statistics."""

import json
from pathlib import Path

from antimeridian import bbox, fix_multi_polygon, fix_polygon
import geopandas as gpd
import pandas as pd
from s3fs import S3FileSystem

from common import (
    get_change_magnitude_summary,
    get_change_type_summary,
    remove_exclusions,
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

    # Summarise km of different roc categories
    ratesofchange = gpd.read_file(
        coastlines_file, layer="rates_of_change", engine="pyogrio", use_arrow=True
    ).rename(dict(eez_territory="ISO_Ter1"), axis=1)

    # Use all significant rates of change data with high certainty.
    # We do _not_ filter by the amount of change at this point
    ratesofchange = remove_exclusions(ratesofchange)
    roc_stats = summarise_roc(ratesofchange.groupby("ISO_Ter1"))

    # Estimate bounding box based on economic exclusion zones
    eez = eez.dissolve(by="ISO_Ter1")

    def fix_and_bbox(geometry):
        fixer = fix_polygon if geometry.geom_type == "Polygon" else fix_multi_polygon
        return bbox(fixer(geometry), force_over_antimeridian=True)

    eez["bbox"] = eez.geometry.to_crs(4326).apply(fix_and_bbox)

    output = (
        # Do right join to add roc stats for countries without hotspots
        hotspot_stats.join(roc_stats, how="right")
        .join(eez[["bbox"]])
        # NaN may exist from right join above
        .fillna(0)
        .reset_index(names="id")
    )
    output.to_csv(OUTPUT_DIR / "country_summaries.csv", index=False)
    _write_geojson(output, OUTPUT_DIR / "country_summaries.geojson")


def _write_geojson(df: pd.DataFrame, output_path: Path) -> None:
    """Write country-level summaries to geojson.

    Format is as requested by front end developers.

    Args:
        df: Country-level stats
        output_path: Where to write the output. Output is also copied to S3.
    """

    features = []
    for _, row in df.iterrows():
        feature = {
            "type": "Feature",
            "bbox": row["bbox"],
            "geometry": {"type": "Point", "coordinates": None},
            "properties": {
                "id": row["id"],
                "shoreline_change_direction": {
                    "retreat_km": row["retreat_km"],
                    "retreat_non_sig_km": row["retreat_non_sig_km"],
                    "growth_km": row["growth_km"],
                    "growth_non_sig_km": row["growth_non_sig_km"],
                    "stable_km": row["stable_km"],
                },
                "shoreline_change_magnitude": {
                    "high_change_km": row["high_change_km"],
                    "medium_change_km": row["medium_change_km"],
                    "low_change_km": row["low_change_km"],
                },
                "population_in_hotspots": row["total_population"],
                "number_of_buildings_in_hotspots": row["building_counts"],
                "mangrove_area_ha_in_hotspots": row["mangrove_area_ha"],
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
        - percent_{growth|retreat|stable|growth_non_sig|retreat_non_sig} : The percent
            of points in the corresponding category. Significance is determined
            by the sig_time column.
        - {high|medium|low|non_sig}_km : The kilometers of coastline in each
            category.
    """

    change_type_summary = roc.apply(
        get_change_type_summary, include_groups=False, summary_type="km"
    ).unstack(fill_value=0)
    change_magnitude_summary = roc.apply(
        get_change_magnitude_summary, include_groups=False, summary_type="km"
    ).unstack(fill_value=0)

    return change_type_summary.join(change_magnitude_summary)


if __name__ == "__main__":
    country_level_stats()
