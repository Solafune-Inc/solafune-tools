import geopandas
import solafune_tools
import pytest
import os

@pytest.fixture()
def get_catalog():
    df_loc = solafune_tools.planetary_computer_stac_query(aoi_geometry_file='tests/data-test/geojson/sample.geojson')
    df = geopandas.read_parquet(df_loc)
    yield df
    os.remove(df_loc)

def test_get_catalog(get_catalog):
    assert isinstance(get_catalog, geopandas.GeoDataFrame)

