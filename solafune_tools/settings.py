import os


def set_data_directory(dir_path=os.path.join(os.getcwd(), "data/")):
    """
    Sets the data directory as an environment variable and creates
    various subdirectories used by this package.
    """
    dir_path = os.path.abspath(dir_path)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
        for each in ["stac", "geojson", "tif", "parquet", "logs", "mosaic"]:
            os.mkdir(os.path.join(dir_path, each))
    os.environ["solafune_tools_data_dir"] = dir_path
    assert os.environ.get("solafune_tools_data_dir") == dir_path
    return None


def get_data_directory():
    """Gets the data directory from the environment variable"""
    if os.getenv("solafune_tools_data_dir") is None:
        set_data_directory()
    return os.getenv("solafune_tools_data_dir")


def get_valid_bands():
    """Returns a list of valid bands for Sentinel-2"""
    return [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ]
