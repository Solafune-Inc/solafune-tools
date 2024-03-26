import glob
import os
from datetime import datetime

import geopandas as gpd
import pystac
import rasterio
import rioxarray
import stac_geoparquet
from shapely.geometry import Polygon, mapping

import solafune_tools.image_fetcher
import solafune_tools.settings

data_dir = solafune_tools.settings.get_data_directory()


def _get_local_filename(tif_dir, band, remote_href):
    """Utility for creating local filenames based on remote filenames"""
    cwd = os.getcwd()
    basename = os.path.basename(remote_href.split(".tif")[0] + ".tif")
    outfile_loc = os.path.join(cwd, tif_dir, band, basename)
    return outfile_loc


def _local_file_links_update(row, bands, tif_dir):
    """Utility for updating local file location paths"""
    new_row = {band: row[band] for band in bands}
    for band in new_row.keys():
        new_row[band]["href"] = _get_local_filename(tif_dir, band, row[band]["href"])
    return new_row


def create_local_catalog_from_existing(
    input_catalog_parquet=os.path.join(
        data_dir, "parquet/2023_May_July_CuCoBounds.parquet"
    ),
    bands=["B04", "B03", "B02"],
    tif_files_dir=os.path.join(data_dir, "tif/"),
    outfile_dir="Auto",
):
    """
    Update a downloaded Planetary Computer STAC catalog geoparquet file into a
    local STAC catalog for stacking and mosaicking using the library stackstac

    Parameters
    ----------
    input_catalog_parquet : str | path
                            Location of downloaded catalog of remote files in
                            parquet format
    bands : list(str)
            list of bands for which files were downloaded
    tif_files_dir : str | path
                    Directory where downloaded files were stored.
    outfile_dir : str | path
                  Location where to store the local stac catalog
                  in json format
    """
    if outfile_dir == "Auto":
        outfile_dir = os.path.join(
            data_dir,
            "stac",
            os.path.splitext(os.path.basename(input_catalog_parquet))[0],
        )
    parq = solafune_tools.image_fetcher.filter_redundant_items(
        gpd.read_parquet(input_catalog_parquet)
    )

    parq.assets = parq.assets.apply(
        _local_file_links_update, bands=bands, tif_dir=tif_files_dir
    )
    catalog = pystac.Catalog(
        id="local",
        description="A local catalog for selected scenes downloaded from a remote repo",
    )
    item_collection = stac_geoparquet.to_item_collection(parq)
    for item in item_collection:
        catalog.add_item(item)
    catalog.normalize_hrefs(outfile_dir)
    try:
        catalog.validate_all()
    except pystac.STACValidationError as e:
        print(e)
    catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)
    return os.path.join(outfile_dir, "catalog.json")


def _get_bbox_and_footprint(raster):
    """Utility for getting bbox and geometry from a raster"""
    with rasterio.open(raster) as r:
        bounds = r.bounds
        bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]
        footprint = Polygon(
            [
                [bounds.left, bounds.bottom],
                [bounds.left, bounds.top],
                [bounds.right, bounds.top],
                [bounds.right, bounds.bottom],
            ]
        )

        return (bbox, mapping(footprint))


def _get_datetime(path):
    """Utility for getting start and end dates from a filename"""
    filename = os.path.basename(path)
    start_date = datetime.strptime(filename[0:10], "%Y_%m_%d")
    end_date = filename[11:21]
    return [start_date, end_date]


def create_local_catalog_from_scratch(
    infile_dir="Auto", outfile_loc="Auto"
) -> os.PathLike:
    """
    Creates a local STAC catalog given a folder of tif files. Primarily
    used to catalog the repository of newly created mosaics.

    Parameters
    ----------
    infile_dir : str | path
                Location of tif file directory to be cataloged. 'Auto' points
                to the mosaic subdir in the data directory.
    outfiledir : str | path
                 Location where catalog will be stored. 'Auto' defaults to
                 'data/stac/mosaic/catalog.json`
    """
    catalog = pystac.Catalog(
        id="mosaics",
        description="This catalog contains median mosaics created from Sentinel-2 data \
        using solafune_tools",
    )
    if infile_dir == "Auto":
        infile_dir = os.path.join(data_dir, "mosaic")
    # files = sorted(glob.glob(os.path.join(infile_dir, "*.tif")))
    file_paths = []
    for path, _, files in os.walk(infile_dir):
        for name in files:
            file_path = os.path.join(path, name)
            if os.path.splitext(file_path)[-1] == ".tif":
                file_paths.append(file_path)
    for img_path in file_paths:
        bbox, footprint = _get_bbox_and_footprint(img_path)

        start_date, end_date = _get_datetime(img_path)
        with rasterio.open(img_path) as r:
            epsg = int(str(r.crs).removeprefix("EPSG:"))
            transform = list(r.transform)
            shape = r.shape
        with rioxarray.open_rasterio(img_path) as ds:
            bands = list(ds.long_name) if type(ds.long_name) == tuple else list([ds.long_name])
        item = pystac.Item(
            id=os.path.splitext(os.path.basename(img_path))[0],
            geometry=footprint,
            bbox=bbox,
            datetime=start_date,
            properties={
                # "epsg": epsg,
                "end_date": end_date,
                "bands": bands,
                "epsg": epsg,
                "transform": transform,
                "shape": shape,
            },
        )

        item.add_asset(
            key="mosaic",
            asset=pystac.Asset(
                href=os.path.abspath(img_path),
                media_type=pystac.MediaType.GEOTIFF,
            ),
        )
        catalog.add_item(item)
    if outfile_loc == "Auto":
        outfile_loc = os.path.abspath(os.path.join(data_dir, "stac", "mosaic"))
    catalog.normalize_hrefs(outfile_loc)
    try:
        catalog.validate_all()
    except pystac.STACValidationError as e:
        print(e)
    catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)
    return os.path.join(outfile_loc, "catalog.json")


def get_catalog_items_as_gdf(
    file_path: str | os.PathLike = os.path.join(data_dir, "stac/catalog.json"),
) -> gpd.GeoDataFrame:
    """Loads a STAC catalog as a geopandas GeoDataFrame"""
    catalog = pystac.Catalog.from_file(file_path)
    # collection = catalog.get_child(id='Sentinel-2')
    item_links = [x.to_dict() for x in catalog.get_items()]
    df = stac_geoparquet.to_geodataframe(item_links)
    return df
