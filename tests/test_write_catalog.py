import solafune_tools.image_fetcher
import solafune_tools.settings
import pytest
import shutil
import random
import string
import os


@pytest.fixture()
def get_catalog():
    random_dir_name = "".join(
        random.choices(string.ascii_lowercase + string.digits, k=6)
    )
    solafune_tools.settings.set_data_directory(random_dir_name)
    parq_loc = solafune_tools.image_fetcher._make_parquet_filename(
        start_date="2023-05-01",
        end_date="2023-08-01",
        aoi_geometry_file="data/geojson/sample.geojson",
    )
    yield [solafune_tools.settings.get_data_directory(), parq_loc]
    shutil.rmtree(solafune_tools.settings.get_data_directory())
    os.environ.pop("solafune_tools_data_dir")


def test_geoparquet_dir(get_catalog):
    data_dir_from_environ = os.path.join(get_catalog[0], "parquet")
    data_dir_from_write = os.path.dirname(get_catalog[1])
    assert data_dir_from_environ == data_dir_from_write
