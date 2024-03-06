import logging
import os
from statistics import mode

import pystac
# even though rioxarray is not explicitly used,
# it is needed for rio.to_raster on xarray dataarray
import rioxarray
import stackstac

import solafune_tools.settings

data_dir = solafune_tools.settings.get_data_directory()


def _get_most_common_epsg(items):
    """Finds the most common crs string from a stack of tif files"""
    epsg_list = []
    for item in items:
        epsg_list.append(item.to_dict()["properties"]["proj:epsg"])
    return mode(epsg_list)


def create_mosaic(
    local_stac_catalog=os.path.join(data_dir, "stac/catalog.json"),
    outfile_loc="Auto",
    out_epsg="Auto",
    resolution=100,
):
    """
    Creates a median mosaic from a STAC catalog given a target epsg
    and output resolution (in the target epsg units, careful of meter
    and degrees units). This function will use a Dask cluster if available,
    and it is highly recommended to use Dask to get results in a
    reasonable amount of time.
    """
    logging.warning(
        "!!! Make sure a Dask server is running and accessible."
        " If not, stop the execution of the mosaicking function and start one !!!"
    )
    catalog = pystac.Catalog.from_file(local_stac_catalog)
    items = list(catalog.get_items(recursive=True))
    if out_epsg == "Auto":
        out_epsg = _get_most_common_epsg(items)

    stack = stackstac.stack(items, epsg=out_epsg, resolution=resolution)
    median = (
        stack.dropna(dim="time", how="all")
        .groupby("band")
        .median(dim="time", skipna=True)
    )
    outval = median.compute()
    if outfile_loc == "Auto":
        outfile_basename = (
            os.path.split(os.path.dirname(local_stac_catalog))[-1] + ".tif"
        )
        outfile_loc = os.path.join(data_dir, "mosaic", outfile_basename)
    if not os.path.isdir(os.path.dirname(outfile_loc)):
        os.mkdir(os.path.dirname(outfile_loc))
    # set band index to names instead of numerical index
    bands = list(items[0].assets.keys())
    outval["band"] = bands
    outval.rio.to_raster(outfile_loc)
    return outfile_loc
