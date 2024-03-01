import os

import pystac
import stac_geoparquet
from geopandas import GeoDataFrame


def get_catalog_items(
    file_path: str | os.PathLike = "/home/pushkar/geodatabase/data/stac/catalog.json",
) -> GeoDataFrame:
    catalog = pystac.Catalog.from_file(file_path)
    # collection = catalog.get_child(id='Sentinel-2')
    item_links = [x.to_dict() for x in catalog.get_items()]
    df = stac_geoparquet.to_geodataframe(item_links)
    return df
