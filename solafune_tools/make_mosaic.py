import os
from statistics import mode

import pystac
import rioxarray
import stackstac
from settings import set_data_directory

if os.getenv("solafune_tools_data_dir") is None:
    set_data_directory()
data_dir = os.getenv("solafune_tools_data_dir")


def get_most_common_epsg(items):
    epsg_list = []
    for item in items:
        epsg_list.append(item.to_dict()["properties"]["proj:epsg"])
    return mode(epsg_list)


def make_mosaic(
    stac_catalog=os.path.join(data_dir, "stac/catalog.json"),
    outfile_loc=os.path.join(data_dir, "outval.tif"),
    epsg=None,
    resolution=100,
):
    catalog = pystac.Catalog.from_file(stac_catalog)
    items = list(catalog.get_items(recursive=True))
    if epsg is None:
        epsg = get_most_common_epsg(items)
    stack = stackstac.stack(items, epsg=epsg, resolution=resolution)
    median = stack.groupby("band").median(dim="time", skipna=True)
    outval = median.compute()
    outval.rio.to_raster(outfile_loc)
    return outfile_loc
