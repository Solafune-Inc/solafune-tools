import os

import solafune_tools


def test_data_directories():
    data_dir = solafune_tools.get_data_directory()
    assert os.path.isdir(data_dir)
    for each in ["stac", "geojson", "tif", "parquet", "logs"]:
        assert os.path.isdir(os.path.join(data_dir, each))
