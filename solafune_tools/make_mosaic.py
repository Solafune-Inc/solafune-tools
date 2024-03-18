import json
import logging
import math
import os
import shutil
from statistics import mode
from importlib.metadata import version

import pystac

# even though rioxarray is not explicitly used,
# it is needed for rio.to_raster on xarray dataarray
import rioxarray
import stackstac
import xarray

import solafune_tools.settings

data_dir = solafune_tools.settings.get_data_directory()


def _get_most_common_epsg(items):
    """Finds the most common crs string from a stack of tif files"""
    epsg_list = []
    for item in items:
        epsg_list.append(item.to_dict()["properties"]["proj:epsg"])
    return mode(epsg_list)


def _write_to_file(bands, dataarray, outfile_loc):
    """Writes raster to file with band name in metadata"""
    ds = xarray.Dataset()
    for count, band in enumerate(bands):
        ds[band] = xarray.DataArray(
            dataarray.isel(band=count),
            dims=("y", "x"),
            coords={
                "x": dataarray.x,
                "y": dataarray.y,
                "spatial_ref": dataarray.spatial_ref,
            },
            attrs={
                "long_name": band,
                "solafune_tools_ver": version("solafune_tools")
            },
        )
    ds = ds.assign_attrs(dataarray.attrs)
    ds.astype("uint16").rio.to_raster(outfile_loc)
    return None


def create_mosaic(
    local_stac_catalog=os.path.join(data_dir, "stac", "catalog.json"),
    aoi_geometry_file=None,
    outfile_loc="Auto",
    out_epsg="Auto",
    resolution=100,
    tile_size=None,
    bands="Auto",
    mosaic_style="Multiband",
):
    """
    Creates a median mosaic from a STAC catalog given a target epsg
    and output resolution (in the target epsg units, careful of meter
    and degrees units). This function will use a Dask cluster if available,
    and it is highly recommended to use Dask to get results in a
    reasonable amount of time.

    Parameters
    ----------
    local_stac_catalog : str | path
                         location of stac catalog with links to downloaded
                         tif files
    aoi_geometry_file : str | path
                        geometry to clip mosaic to, defaults to None
    outfile_loc : str | path
                  location where to write out the mosiac. 'Auto' will write to
                  the mosaic subdir in the data directory. If tile_size is not
                  None, this loc becomes a directory where the tiles are stored.
    out_epsg : int
              crs value for the output mosaic. 'Auto' defaults to the most
              common epsg value among the input files.
    resolution : resolution per pixel for the output data. This is in units of the
                crs. For Sentinel-2 data, this is meters. If you use epsg:4326, it
                will be degrees so be careful.
    tile_size : int
                Size of square tiles in pixels for mosiac output. If none, a
                single large tif file is written out.
    bands : str | list(str)
           Pass in a list of bands for which you need mosaics. If Auto, all bands
           are used.
    mosaic_style : 'Singleband' | 'Multiband''
                    Whether a single multiband mosaic is needed or individual
                    mosaics for every band

    """
    logging.warning(
        "!!! Make sure a Dask server is running and accessible."
        " If not, stop the execution of the mosaicking function and start one !!!"
    )
    catalog = pystac.Catalog.from_file(local_stac_catalog)
    items = list(catalog.get_items(recursive=True))
    if out_epsg == "Auto":
        out_epsg = _get_most_common_epsg(items)

    if bands == "Auto":
        bands = list(items[0].assets.keys())

    stack = stackstac.stack(items, epsg=out_epsg, resolution=resolution)
    median = (
        stack.dropna(dim="time", how="all")
        .sel(band=bands)
        .groupby("band")
        .median(dim="time", skipna=True)
    )

    if aoi_geometry_file != None:
        with open(aoi_geometry_file) as f:
            data = json.load(f)
        area_of_interest = data["features"][0]["geometry"]
        median = median.rio.clip(geometries=[area_of_interest], crs=4326)
    
    catalog_basename = os.path.split(os.path.dirname(local_stac_catalog))[-1]

    if tile_size == None:
        outval = median.compute()
        if outfile_loc == "Auto":
            outfile_basename = (
                catalog_basename
                + "_".join(bands)
                + ".tif"
            )
            outfile_loc = os.path.join(data_dir, "mosaic", outfile_basename)
        if not os.path.isdir(os.path.dirname(outfile_loc)):
            os.mkdir(os.path.dirname(outfile_loc))
        # set band index to names instead of numerical index
        _write_to_file(bands=bands, dataarray=outval, outfile_loc=outfile_loc)
        # outval.astype('uint16').rio.to_raster(outfile_loc)
        return outfile_loc

    else:
        n_x_tiles = math.ceil(len(median.x) / tile_size)
        n_y_tiles = math.ceil(len(median.y) / tile_size)

        if outfile_loc == "Auto":
            outdir_loc = os.path.join(
                data_dir,
                "mosaic",
                catalog_basename,
            )
        else:
            outdir_loc = outfile_loc

        if os.path.isdir(outdir_loc):
            shutil.rmtree(outdir_loc)

        os.mkdir(outdir_loc)
        if mosaic_style == "Multiband":
            for i in range(n_x_tiles):
                for j in range(n_y_tiles):
                    tile_data = median.sel(
                        x=median.x[i * tile_size : (i + 1) * tile_size],
                        y=median.y[j * tile_size : (j + 1) * tile_size],
                    )
                    # tile_data["band"] = bands
                    band_ids = "_".join(bands)
                    tile_file_loc = os.path.join(
                        outdir_loc,
                        f"{catalog_basename}_{band_ids}_tile_{i+1}_{j+1}.tif",
                    )
                    _write_to_file(
                        bands=bands, dataarray=tile_data, outfile_loc=tile_file_loc
                    )
        elif mosaic_style == "Singleband":
            for i in range(n_x_tiles):
                for j in range(n_y_tiles):
                    tile_data = median.sel(
                        x=median.x[i * tile_size : (i + 1) * tile_size],
                        y=median.y[j * tile_size : (j + 1) * tile_size],
                    )
                    # tile_data["band"] = bands
                    for band in bands:
                        tile_file_loc = os.path.join(
                            outdir_loc,
                            f"{catalog_basename}_{band}_tile_{i+1}_{j+1}.tif",
                        )
                        _write_to_file(
                            bands=[band], dataarray=tile_data, outfile_loc=tile_file_loc
                        )
        else:
            raise ValueError(
                "Please use either 'Multiband' or 'Singleband' \
                         only for the parameter 'mosaic_style'"
            )

        return outdir_loc
