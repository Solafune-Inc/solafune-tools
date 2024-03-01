from geopandas import GeoDataFrame

import solafune_tools


def test_get():
    df = solafune_tools.get_catalog_items()
    assert isinstance(df, GeoDataFrame)
