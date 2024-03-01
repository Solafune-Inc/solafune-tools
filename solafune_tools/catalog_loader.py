import os

import pystac
import stac_geoparquet
from geopandas import GeoDataFrame
from settings import set_data_directory

if os.getenv("solafune_tools_data_dir") is None:
    set_data_directory()
data_dir = os.getenv("solafune_tools_data_dir")


def get_catalog_items(
    file_path: str | os.PathLike = os.path.join(data_dir, "stac/catalog.json"),
) -> GeoDataFrame:
    catalog = pystac.Catalog.from_file(file_path)
    # collection = catalog.get_child(id='Sentinel-2')
    item_links = [x.to_dict() for x in catalog.get_items()]
    df = stac_geoparquet.to_geodataframe(item_links)
    return df
