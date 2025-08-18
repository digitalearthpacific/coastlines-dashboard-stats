import geopandas as gpd
import numpy as np
import pandas as pd


def categorize_change_magnitude(roc, no_change_threshold: int = 2):
    sig = _sig_change(roc)
    rules = [
        (sig & (roc.rate_time > 5), "high_change"),
        (sig & (roc.rate_time > 3), "medium_change"),
        (sig & (roc.rate_time > no_change_threshold), "low_change"),
    ]
    conditions, categories = zip(*rules)
    return pd.Series(
        np.select(conditions, categories, default="non_sig"), index=roc.index
    )


def categorize_roc(roc, no_change_threshold: int = 2):
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


def change_magnitude_percentages(roc: gpd.GeoDataFrame):
    return _calculate_percent_of_each_value(categorize_change_magnitude(roc))


def change_type_percentages(roc: gpd.GeoDataFrame, no_change_threshold: int = 2):
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
    return _calculate_percent_of_each_value(categorize_roc(roc, no_change_threshold))
