import json
from pathlib import Path

from antimeridian import bbox, fix_multi_polygon, fix_polygon
import geopandas as gpd
import pandas as pd
from s3fs import S3FileSystem

from common import (
    categorize_change_direction,
    categorize_change_magnitude,
    get_change_magnitude_summary,
    get_change_type_summary,
)
from config import COASTLINES_FILE, EEZ, OUTPUT_DIR, VERSION


def summarise_roc(roc: pd.api.typing.DataFrameGroupBy) -> gpd.GeoDataFrame:
    def get_distance_stats(d: pd.DataFrame):
        return (
            d.loc[
                d.certainty == "good",
                d.columns.str.contains("dist_|ISO"),
            ]
            .median()
            .round(2)
        )

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


# I don't think this will be needed because as of right now we just
# need
def country_level_stats(
    coastlines_file: Path = COASTLINES_FILE, eez: gpd.GeoDataFrame = EEZ
):
    # Sum population, buildings, and mangrove area in all contiguous hotspots
    # in the entire country
    hotspot_stats = (
        gpd.read_file("data/output/contiguous_hotspots.gpkg")
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

    output = (
        hotspot_stats.join(roc_stats)
        .join(eez[["bbox"]])
        .reset_index()
        .rename(dict(ISO_Ter1="id"), axis=1)
    )
    output.to_csv(OUTPUT_DIR / "country_summaries.csv", index=False)
    _write_geojson(output, OUTPUT_DIR / "country_summaries.geojson")


def contiguous_hotspots_groups_stats():
    contiguous_hotspots = gpd.read_file("data/output/contiguous_hotspots.gpkg")

    # TODO: I _think_ this stands (since hotspots were made from good points)
    contiguous_hotspots["certainty"] = "good"
    # Create categorization for combined hotspot values
    contiguous_hotspots["change_magnitude"] = categorize_change_magnitude(
        contiguous_hotspots
    )
    contiguous_hotspots["change_direction"] = categorize_change_direction(
        contiguous_hotspots
    )

    output = summarise_roc(
        contiguous_hotspots.groupby(
            ["ISO_Ter1", "change_magnitude", "change_direction"]
        )
    )
    output.reset_index().to_csv(OUTPUT_DIR / "country_hotspot_group_summaries.csv")


def _write_geojson(df: pd.DataFrame, output_path: Path) -> None:
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
                    str(year): row[f"dist_{year}"] for year in range(1999, 2024)
                },
            },
        }
        features.append(feature)

    geojson_obj = {"type": "FeatureCollection", "features": features}

    with open(output_path, "w") as f:
        json.dump(geojson_obj, f, indent=2)

    s3 = S3FileSystem()
    s3_path = f"dep-public-staging/dep_ls_coastlines/dashboard_stats/{VERSION.replace('.', '-')}"
    s3.put(output_path, s3_path)


def main():
    country_level_stats()


if __name__ == "__main__":
    main()
