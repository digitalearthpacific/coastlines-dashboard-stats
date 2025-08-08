# Not finished, and sparkgeo will probably do this anyway
def captial_cities():
    capitals = gpd.read_file(
        "https://www.naturalearthdata.com/http//www.naturalearthdata.com/download/10m/cultural/ne_10m_populated_places.zip"
    ).query("@FEATURECLA == 'Admin-0 captial'")
