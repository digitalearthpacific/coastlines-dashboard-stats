import geohash as gh
import geopandas as gpd

from coastlines.vector import change_regress


def calculate_rates_of_change_over_polygons(
    polygons: gpd.GeoDataFrame, ratesofchange: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Calculate combined rates of change values for each of the given polygons.

    Take the median distance values for any rates of change points within each
    polygon with "good" certainty values. Use
    :py:func:`coastlines.vector.change_regress` to calculate regression statistics
    across the median values (similar to how hotspot values are calculated).

    Args:
        polygons: A GeoDataFrame with polygons.
        ratesofchange: A rates of change GeoDataFrame

    Returns:
        The input polygons with additional columns rate_time, incpt_time, sig_time,
        se_time, outl_time, and uid,
    """
    # NOTE: There are two approaches if we wanted to remove outliers.
    # First, we could remove outliers in the input rates of change tables
    # before calculating stats (and establishing new outliers), or we could
    # take the median of all, calculate stats, and remove outliers there.
    # Or we could do both. Currently we're doing neither, because I'm not
    # sure what the "dist_" stats are being used for (if anything).
    # 12Sept25 -> they've been removed from anything.

    # Taken from hotspot calculations in continental.py. It's possible
    # they're close enough we could have a single function, but there
    # are a few differences.
    # Spatial join rate of change points to each polygon
    hotspot_grouped = (
        ratesofchange.loc[
            ratesofchange.certainty == "good",
            ratesofchange.columns.str.contains("dist_|geometry"),
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
    hotspot_values[["rate_time", "incpt_time", "sig_time", "se_time", "outl_time"]] = (
        rate_out
    )

    # Join aggregated values back to hotspot points after
    # dropping unused columns (regression intercept)
    polygons = gpd.GeoDataFrame(
        polygons.join(hotspot_values.drop("incpt_time", axis=1))
    )

    # Generate a geohash UID for each point and set as index
    uids = (
        # collapsing to centroid here
        polygons.geometry.centroid.to_crs("EPSG:4326")
        .apply(lambda x: gh.encode(x.y, x.x, precision=11))
        .rename("uid")
    )
    return gpd.GeoDataFrame(polygons.set_index(uids))
