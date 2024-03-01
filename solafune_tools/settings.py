import os


def set_data_directory(dir_path=os.path.join(os.getcwd(), "data/")):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
        for each in ["stac", "geojson", "tif", "parquet", "logs"]:
            os.mkdir(os.path.join(dir_path, each))
    os.environ["solafune_tools_data_dir"] = dir_path
    assert os.environ.get("solafune_tools_data_dir") == dir_path
    return None
