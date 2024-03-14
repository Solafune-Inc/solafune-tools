# import geopandas
import time

import planetary_computer
import pystac_client
import pytest

import solafune_tools

# import os

# @pytest.fixture()
# def get_catalog():
#     df_loc = solafune_tools.planetary_computer_stac_query(aoi_geometry_file='tests/data-test/geojson/sample.geojson')
#     df = geopandas.read_parquet(df_loc)
#     yield df
#     os.remove(df_loc)

# def test_get_catalog(get_catalog):
#     assert isinstance(get_catalog, geopandas.GeoDataFrame)
# def make_file(filename: str) -> None:
#     """
#     Function to create a file
#     :param filename: Name of the file to create
#     :return: None
#     """
#     with open(f"{filename}", "w") as f:
#         f.write("hello")

# def test_make_file_with_mock(mocker):
#     """
#     Function to test make file with mock
#     :param mocker: pytest-mock fixture
#     :return: None
#     """
#     filename = "delete_me.txt"

#     # Mock the 'open' function call to return a file object.
#     mock_file = mocker.mock_open()
#     mocker.patch("builtins.open", mock_file)
#     # Call the function that creates the file.
#     make_file(filename)

#     # Assert that the 'open' function was called with the expected arguments.
#     mock_file.assert_called_once_with(filename, "w")

#     # Assert that the file was written to with the expected text.
#     mock_file().write.assert_called_once_with("hello")


# @pytest.fixture()
def get_catalog(junk="Auto"):
    if junk:
        pass
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    time.time()
    return catalog


# def test_get_catalog(get_catalog):
#     assert isinstance(get_catalog, pystac_client.Client)


def test_get_catalog_with_mock(mocker):
    # mock_catalog = mocker.mock_open()
    mocker.patch("pystac_client.Client.open")  # , mock_catalog)
    mocker.patch("time.time")
    get_catalog(junk="what")
    # mock_catalog.assert_called_once_with("https://planetarycomputer.microsoft.com/api/stac/v1",
    #     modifier=planetary_computer.sign_inplace,)
    pystac_client.Client.open.assert_called_once()
    time.time.assert_called_once()
