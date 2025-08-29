"""A quick viewer for country-level distances for debugging purposes."""

import pandas as pd
import plotly.express as px

from config import OUTPUT_DIR


def make_year_plot_viewer():
    country_stats = pd.read_csv(OUTPUT_DIR / "country_summaries.csv")
    dist_columns = country_stats.columns[country_stats.columns.str.contains("dist_")]

    stats_long = country_stats.melt(id_vars="id", value_vars=dist_columns)
    stats_long["year"] = stats_long["variable"].str.replace("dist_", "").astype(int)
    stats_long = stats_long.sort_values(["id", "year"])

    plot = px.line(stats_long, x="year", y="value", color="id")
    buttons = []
    for country in stats_long.id.unique():
        buttons.append(
            dict(
                label=country,
                method="update",
                args=[
                    {"visible": [trace.name == country for trace in plot.data]},
                    {"title": country},
                ],
            )
        )

    initial_country = stats_long.id[0]
    plot.for_each_trace(
        lambda trace: trace.update(visible=trace.name == initial_country)
    )

    plot.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.1,
                "y": 1.2,
            }
        ],
    )

    plot.write_html("docs/line_plot.html")


if __name__ == "__main__":
    make_year_plot_viewer()
