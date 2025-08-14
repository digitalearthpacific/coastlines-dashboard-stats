import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyogrio import read_dataframe


COASTLINES_FILE = Path("data/dep_ls_coastlines_0-7-0-55.gpkg")

eez_file = Path("data/country_boundary_eez.geojson")
if not eez_file.exists():
    read_dataframe(
        "https://pacificdata.org/data/dataset/964dbebf-2f42-414e-bf99-dd7125eedb16/resource/dad3f7b2-a8aa-4584-8bca-a77e16a391fe/download/country_boundary_eez.geojson"
    ).to_file(eez_file)
eez = gpd.read_file(eez_file).to_crs(3832)


def main(coastlines_file: Path = COASTLINES_FILE):
    hotspot_stats = (
        gpd.read_file("data/contiguous_hotspots.gpkg")
        .groupby("ISO_Ter1")[
            ["total_population", "building_counts", "mangrove_area_ha"]
        ]
        .sum()
    )
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
    change_categories = categorize_roc(ratesofchange)
    eez["bbox"] = eez.geometry.to_crs(4326).bounds.apply(list, axis=1)

    output = (
        hotspot_stats.join(ratesofchange_stats)
        .join(change_categories)
        .join(eez.set_index("ISO_Ter1")[["bbox"]])
        .reset_index()
        .rename(dict(ISO_Ter1="id"), axis=1)
    )
    write_geojson(output, Path("data/country_summaries.geojson"))


def categorize_roc(roc: gpd.GeoDataFrame):
    sig = (roc["certainty"].eq("good")) & (roc["sig_time"] < 0.01)
    neg = roc["rate_time"] < 0
    pos = roc["rate_time"] > 0

    rules = [
        (sig & neg, "retreat"),
        (neg, "retreat_non_sig"),
        (sig & pos, "growth"),
        (pos, "growth_non_sig"),
    ]
    conditions, categories = zip(*rules)
    roc["change_category"] = np.select(conditions, categories, default="stable")
    return (
        (roc.groupby("ISO_Ter1")["change_category"].value_counts(normalize=True) * 100)
        .unstack()
        .rename(lambda col: f"percent_{col}", axis=1)
    )


def write_geojson(df: pd.DataFrame, output_path: Path) -> None:

    # Build GeoJSON features
    features = []
    for _, row in df.iterrows():
        feature = {
            "type": "Feature",
            "bbox": row["bbox"],
            "geometry": {"type": "Point", "coordinates": None},
            "properties": {
                "id": row["id"],
                "shoreline_change": {
                    "percent_retreat": row["percent_retreat"],
                    "percent_retreat_non_sig": row["percent_retreat_non_sig"],
                    "percent_growth": row["percent_growth"],
                    "percent_growth_non_sig": row["percent_growth_non_sig"],
                    "percent_stable": row["percent_stable"],
                },
                "population_in_hotspots": row["total_population"],
                "number_of_buildings_in_hotspots": row["building_counts"],
                "mangrove_area_ha_in_hotspots": row["mangrove_area_ha"],
                "median_distances": {
                    "1999": row["dist_1999"],
                    "2000": row["dist_2000"],
                    "2001": row["dist_2001"],
                    "2002": row["dist_2002"],
                    "2003": row["dist_2003"],
                    "2004": row["dist_2004"],
                    "2005": row["dist_2005"],
                    "2006": row["dist_2006"],
                    "2007": row["dist_2007"],
                    "2008": row["dist_2008"],
                    "2009": row["dist_2009"],
                    "2010": row["dist_2010"],
                    "2011": row["dist_2011"],
                    "2012": row["dist_2012"],
                    "2013": row["dist_2013"],
                    "2014": row["dist_2014"],
                    "2015": row["dist_2015"],
                    "2016": row["dist_2016"],
                    "2017": row["dist_2017"],
                    "2018": row["dist_2018"],
                    "2019": row["dist_2019"],
                    "2020": row["dist_2020"],
                    "2021": row["dist_2021"],
                    "2022": row["dist_2022"],
                    "2023": row["dist_2023"],
                },
            },
        }
        features.append(feature)

    # Wrap in FeatureCollection
    geojson_obj = {"type": "FeatureCollection", "features": features}

    # Save to file
    with open(output_path, "w") as f:
        json.dump(geojson_obj, f, indent=2)


if __name__ == "__main__":
    main()
