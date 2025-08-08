from pathlib import Path

import geopandas as gpd


def contiguous_hotspots(hotspots: gpd.GeoDataFrame):
    buffered_hotspots = hotspots[
        (hotspots.sig_time < 0.01) & (hotspots.certainty == "good")
    ][["geometry"]].copy()

    radius = hotspots.radius_m.iloc[0]
    buffered_hotspots["geometry"] = buffered_hotspots.geometry.buffer(radius)

    breakpoint()

    output = gpd.GeoDataFrame(
        geometry=[buffered_hotspots.geometry.union_all()],
        crs=hotspots.crs,
    ).explode(ignore_index=True)
