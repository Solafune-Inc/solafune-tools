import stackstac
import pystac
import rioxarray
from statistics import mode

def get_most_common_epsg(items):
    epsg_list = []
    for item in items:
        epsg_list.append(item.to_dict()['properties']['proj:epsg'])
    return mode(epsg_list)

def make_mosaic(
    stac_catalog="../data/stac/catalog.json",
    outfile_loc="../data/outval.tif",
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