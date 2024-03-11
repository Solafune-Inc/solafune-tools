import logging
import os
import shutil
from statistics import mode
import json
import pystac

# even though rioxarray is not explicitly used,
# it is needed for rio.to_raster on xarray dataarray
import rioxarray
import stackstac
import math
import solafune_tools.settings

data_dir = solafune_tools.settings.get_data_directory()


def _get_most_common_epsg(items):
    """Finds the most common crs string from a stack of tif files"""
    epsg_list = []
    for item in items:
        epsg_list.append(item.to_dict()["properties"]["proj:epsg"])
    return mode(epsg_list)


def create_mosaic(
    local_stac_catalog=os.path.join(data_dir, "stac", "catalog.json"),
    aoi_geometry_file=None,
    outfile_loc="Auto",
    out_epsg="Auto",
    resolution=100,
    tile_size=None,
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

    if aoi_geometry_file != None:
        print(aoi_geometry_file, 'IS NOT NONE')
        with open(aoi_geometry_file) as f:
            data = json.load(f)
        area_of_interest = data["features"][0]["geometry"]
        median = median.rio.clip(geometries=[area_of_interest], crs=4326)
        
    bands = list(items[0].assets.keys())

    if tile_size == None:
        outval = median.compute()
        if outfile_loc == "Auto":
            outfile_basename = (
                os.path.split(os.path.dirname(local_stac_catalog))[-1] + ".tif"
            )
            outfile_loc = os.path.join(data_dir, "mosaic", outfile_basename)
        if not os.path.isdir(os.path.dirname(outfile_loc)):
            os.mkdir(os.path.dirname(outfile_loc))
        # set band index to names instead of numerical index
        outval["band"] = bands
        outval.astype('uint16').rio.to_raster(outfile_loc)
        return outfile_loc

    else:
        n_x_tiles = math.ceil(len(median.x) / tile_size)
        n_y_tiles = math.ceil(len(median.y) / tile_size)

        if outfile_loc == "Auto":
            catalog_basename = os.path.split(os.path.dirname(local_stac_catalog))[-1]
            outdir_loc = os.path.join(
                data_dir,
                "mosaic",
                catalog_basename,
            )
            
        if os.path.isdir(outdir_loc):
            shutil.rmtree(outdir_loc)

        os.mkdir(outdir_loc)

        for i in range(n_x_tiles):
            for j in range(n_y_tiles):
                tile_data = median.sel(
                    x=median.x[i * tile_size : (i + 1) * tile_size],
                    y=median.y[j * tile_size : (j + 1) * tile_size],
                )
                tile_data["band"] = bands
                tile_file_loc = os.path.join(outdir_loc, f"{catalog_basename}_tile_{i+1}_{j+1}.tif")
                tile_data.astype('uint16').rio.to_raster(tile_file_loc)

        return outdir_loc
