import geopandas as gpd
import numpy as np
import pandas as pd

from src.config import CHANGE_THRESHOLD_KM_PER_YR, EXCLUSIONS


def remove_exclusions(
    input_features: gpd.GeoDataFrame, exclusions: gpd.GeoDataFrame = EXCLUSIONS
) -> gpd.GeoDataFrame:
    """Remove features that are in the exclusions dataset.

    Args:
        input_features: Any GeoDataFrame.
        exclusions: Polygon areas to remove from `input_features`.

    Returns:
        The input features with features that are in `exclusions` removed.

    """
    masked = input_features.geometry.apply(lambda geom: exclusions.contains(geom).any())
    return input_features.loc[~masked]


def make_outliers_nan(row: pd.Series):
    """Recode outlier values to nan.

    Args:
        row: Usually a row of rates of change data for coastlines. Must have
        row.dist_{year} properties and a row.outl_time property. The latter
        is a list of zero or more comma separated years.

    Returns: The input with row.dist_{year} set to nan where {year} is coded as an
    outlier.

    """
    for outlier_year in row.outl_time.split(" "):
        if outlier_year != "":
            row[f"dist_{outlier_year}"] = float("nan")
    return row


def categorize_change_magnitude(
    roc, no_change_threshold: int | float = CHANGE_THRESHOLD_KM_PER_YR
):
    """

    Args:
        roc: rates of change
        no_change_threshold: The threshold under which no change is assumed to
            have occurred.

    Returns:

    """
    sig = _sig_change(roc)
    rules = [
        (sig & (abs(roc.rate_time) > 5), "high_change"),
        (sig & (abs(roc.rate_time) > 3), "medium_change"),
        (sig & (abs(roc.rate_time) > no_change_threshold), "low_change"),
    ]
    conditions, categories = zip(*rules)
    return pd.Series(
        np.select(conditions, categories, default="non_sig"), index=roc.index
    )


def categorize_change_direction(
    roc, no_change_threshold: int | float = CHANGE_THRESHOLD_KM_PER_YR
):
    sig = _sig_change(roc)
    neg = roc["rate_time"] < -no_change_threshold
    pos = roc["rate_time"] > no_change_threshold

    rules = [
        (sig & neg, "retreat"),
        (neg, "retreat_non_sig"),
        (sig & pos, "growth"),
        (pos, "growth_non_sig"),
    ]
    conditions, categories = zip(*rules)
    return pd.Series(
        np.select(conditions, categories, default="stable"), index=roc.index
    )


def _sig_change(roc) -> pd.Series:
    return (roc["certainty"].eq("good")) & (roc["sig_time"] < 0.01)


def _calculate_percent_of_each_value(series: pd.Series):
    output = series.value_counts(normalize=True) * 100
    output.index = [f"percent_{col}" for col in output.index]
    return output.round(2)


def _calculate_km_of_each_value(series: pd.Series):
    roc_interval_m = 30
    km_per_m = 1 / 1000
    roc_interval_km = roc_interval_m * km_per_m

    output = series.value_counts() * roc_interval_km
    output.index = [f"{col}_km" for col in output.index]
    return output.round(2)


def _calculate_cumulative_km_of_each_value(series: pd.Series) -> pd.Series:
    km_of_each = _calculate_km_of_each_value(series)
    # This syntax is needed to fill missings. I tried to index with fill=0 but
    # the join didn't work afterwards and everything looked exactly the same
    km_of_each["medium_change_km"] = km_of_each.get(
        "high_change_km", 0
    ) + km_of_each.get("medium_change_km", 0)

    # medium now includes medium + high
    km_of_each["low_change_km"] = km_of_each.get("low_change_km", 0) + km_of_each.get(
        "medium_change_km", 0
    )
    return km_of_each.round(2)


def get_change_magnitude_summary(
    roc: gpd.GeoDataFrame,
    no_change_threshold: int | float = CHANGE_THRESHOLD_KM_PER_YR,
    summary_type="cumulative_km",
):
    match summary_type:
        case "km":
            summariser = _calculate_km_of_each_value
        case "cumulative_km":
            summariser = _calculate_cumulative_km_of_each_value
        case _:
            summariser = _calculate_percent_of_each_value
    return summariser(categorize_change_magnitude(roc, no_change_threshold))


def get_change_type_summary(
    roc: gpd.GeoDataFrame,
    no_change_threshold: int | float = CHANGE_THRESHOLD_KM_PER_YR,
    summary_type="km",
):
    """Categories rates of change points into growth, retreat, and stable classes.

    Args:
        roc: Rates of change points, with at least "certainty", "sig_time" and
            "rate_time" columns. Summary stats will be tabulated by groups if
            input is grouped
        no_change_threshold:
            The absolute value of the positive and negative cutoff for the "stable"
            class, symmetric around zero.

    Returns:
        The input dataframe with additional columns.

    """
    summariser = (
        _calculate_percent_of_each_value
        if summary_type == "percent"
        else _calculate_km_of_each_value
    )
    return summariser(categorize_change_direction(roc, no_change_threshold))
