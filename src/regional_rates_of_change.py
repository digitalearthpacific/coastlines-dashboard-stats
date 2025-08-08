def calculate_rates_of_change_over_polygons(polygons, ratesofchange_gdf):
    # Taken from hotspot calculations in continental.py. It's possible
    # they're close enough we could have a single function, but there
    # are a few differences.
    # Spatial join rate of change points to each polygon
    hotspot_grouped = (
        ratesofchange_gdf.loc[
            ratesofchange_gdf.certainty == "good",
            ratesofchange_gdf.columns.str.contains("dist_|geometry"),
        ]
        .sjoin(polygons, predicate="within")
        .drop(columns=["geometry"])
        .groupby("index_right")
    )

    # Aggregate/summarise values by taking median of all points
    # within each buffered polygon
    hotspot_values = hotspot_grouped.median().round(2)

    # Extract year from distance columns (remove "dist_")
    x_years = hotspot_values.columns.str.replace("dist_", "").astype(int)

    # Compute coastal change rates by linearly regressing annual
    # movements vs. time
    rate_out = hotspot_values.apply(
        lambda row: change_regress(
            y_vals=row.values.astype(float), x_vals=x_years, x_labels=x_years
        ),
        axis=1,
    )

    # Add rates of change back into dataframe
    polygons[["rate_time", "incpt_time", "sig_time", "se_time", "outl_time"]] = rate_out

    # Join aggregated values back to hotspot points after
    # dropping unused columns (regression intercept)
    polygons = polygons.join(hotspot_values.drop("incpt_time", axis=1))

    # Generate a geohash UID for each point and set as index
    uids = (
        hotspots_gdf.geometry.to_crs("EPSG:4326")
        .apply(lambda x: gh.encode(x.y, x.x, precision=11))
        .rename("uid")
    )
    hotspots_gdf = hotspots_gdf.set_index(uids)
