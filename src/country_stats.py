import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from common import get_change_magnitude_percentages, get_change_type_percentages
from config import COASTLINES_FILE, EEZ, OUTPUT_DIR


def summarise_roc(roc: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    stats = (
        roc["total_population", "building_counts", "mangrove_area_ha"].sum().round(2)
    )
    change_type_percentages = roc.apply(
        get_change_type_percentages, include_groups=False
    ).unstack(fill_value=0)
    change_magnitude_percentages = roc.apply(
        get_change_magnitude_percentages, include_groups=False
    ).unstack(fill_value=0)

    return (
        stats.join(ratesofchange_stats)
        .join(change_type_percentages_by_country)
        .join(change_magnitude_percentages_by_country)
    )


def main(coastlines_file: Path = COASTLINES_FILE, eez: gpd.GeoDataFrame = EEZ):
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
    ratesofchange_stats = (
        ratesofchange.loc[
            ratesofchange.certainty == "good",
            ratesofchange.columns.str.contains("dist_|ISO"),
        ]
        .groupby("ISO_Ter1")
        .median()
        .round(2)
    )

    # Estimate bounding box based on
    eez["bbox"] = eez.geometry.to_crs(4326).bounds.apply(list, axis=1)

    change_type_percentages_by_country = (
        ratesofchange.groupby("ISO_Ter1")
        .apply(change_type_percentages, include_groups=False)
        .unstack(fill_value=0)
    )
    change_magnitude_percentages_by_country = (
        ratesofchange.groupby("ISO_Ter1")
        .apply(change_magnitude_percentages, include_groups=False)
        .unstack(fill_value=0)
    )

    output = (
        hotspot_stats.join(ratesofchange_stats)
        .join(change_type_percentages_by_country)
        .join(change_magnitude_percentages_by_country)
        .join(eez.set_index("ISO_Ter1")[["bbox"]])
        .reset_index()
        .rename(dict(ISO_Ter1="id"), axis=1)
    )
    breakpoint()
    _write_geojson(output, OUTPUT_DIR / "country_summaries.geojson")


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
                    "percent_high_change": row["percent_high_change"],
                    "percent_medium_change": row["percent_medium_change"],
                    "percent_low_change": row["percent_low_change"],
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


if __name__ == "__main__":
    main()
