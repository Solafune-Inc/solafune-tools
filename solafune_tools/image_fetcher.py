import json
import logging
import os
import sys
import time

import geopandas as gpd
import PIL
import planetary_computer
import pystac_client
import requests
import stac_geoparquet
from PIL import Image

import solafune_tools.settings

logging.basicConfig(level=logging.INFO)
PIL.Image.MAX_IMAGE_PIXELS = 2e8

data_dir = solafune_tools.settings.get_data_directory()


def _log(func):
    """Decorator that logs the runtime of a function"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} {result} time {end_time-start_time:.2f}s")
        return result

    return wrapper


def _check_downloaded_file(file_path, raw_file_name) -> bool:
    """Function that opens a downloaded tif file to check if it is uncorrupted"""
    try:
        Image.open(file_path)
    except Exception as e:
        with open(os.path.join(data_dir, "logs/bad_responses.txt"), "a") as logfile:
            logfile.write(f"{os.path.basename(file_path)} {raw_file_name} {e}\n")
        return False
    return True


@_log
def _download_tiff(raw_file_name, outfile_dir, session) -> int:
    """Downloads a single file given a filename and destination directory"""
    dest_file_loc = os.path.join(outfile_dir, os.path.basename(raw_file_name))
    if os.path.isfile(dest_file_loc) and _check_downloaded_file(
        dest_file_loc, raw_file_name
    ):
        return 200
    sample = planetary_computer.sign(raw_file_name)
    response = session.get(sample, stream=True)
    if response.ok:
        with open(dest_file_loc, "wb") as dest_file:
            for chunk in response.iter_content(chunk_size=10 * 1024):
                dest_file.write(chunk)
    else:
        with open(os.path.join(data_dir, "logs/bad_responses.txt"), "a") as logfile:
            logfile.write(f"{response.status_code} {raw_file_name}\n")

        if response.status_code == 429:
            sys.exit("too many requests")
    _check_downloaded_file(dest_file_loc, raw_file_name)
    print(dest_file_loc)
    return response.status_code


def _setup_directories(outfile_dir, bands) -> None:
    """Sets up subdirectories for each band in the download directory"""
    if not os.path.isdir(outfile_dir):
        os.mkdir(outfile_dir)
    for band in bands:
        band_dir = os.path.join(outfile_dir, band)
        if not os.path.isdir(band_dir):
            os.mkdir(band_dir)
    return None


def filter_redundant_items(dataframe) -> gpd.GeoDataFrame:
    """
    Filtering function to reduce the number of files to be downloaded.
    Need to add few more conditions, currently only one.
    """
    reduce_df = (
        dataframe.sort_values(by=["s2:nodata_pixel_percentage", "eo:cloud_cover"])
        .groupby(["geometry"])
        .head(5)
    )
    """
    count_index = rdf.geometry.map(rdf.geometry.value_counts()) >= 5
    suff_count_shape = rdf.geometry.loc[count_index].unary_union
    shape_index = np.logical_not(np.isclose(rdf.geometry.union(suff_count_shape).area, suff_count_shape.area, rtol=5e-2))
    total_index = count_index | shape_index
    
    """
    return reduce_df


@_log
def planetary_computer_fetch_images(
    dataframe_path=os.path.join(data_dir, "parquet/2023_May_July_CuCoBounds.parquet"),
    bands=["B02", "B03", "B04"],
    outfile_dir="Auto",
) -> os.PathLike:
    """
    Iterates over assets in a catalog in geoparquet file and downloads
    selected bands.

    Parameters
    ----------
    dataframe_path : str | path
                     location of the stac catalog parquet file
    bands : list(str)
            selection of bands to download
    outfile_dir : str | path
                  location where to write out the downloaded tif files.
                  'Auto' will write to the tif subdir in the data directory.
    """
    gdf = filter_redundant_items(gpd.read_parquet(dataframe_path))
    assets = gdf.assets
    if outfile_dir == "Auto":
        outfile_dir = os.path.join(
            data_dir, "tif", os.path.splitext(os.path.basename(dataframe_path))[0]
        )
    _ = _setup_directories(outfile_dir, bands)
    total = len(gdf.assets)
    session = requests.Session()
    for count, each in enumerate(assets):
        print(f"Downloading file {count+1} of {total}")
        for band in bands:
            print(f"Downloading {band}")
            get_file = each[band]["href"]
            _download_tiff(
                raw_file_name=get_file.split(".tif")[0] + ".tif",
                outfile_dir=os.path.join(outfile_dir, band),
                session=session,
            )

    return outfile_dir


def _make_parquet_filename(start_date, end_date, aoi_geometry_file) -> os.PathLike:
    """
    Creates a filename including metadata to store a stac catalog
    as a geoparquet file.
    """
    base_geometry_name = os.path.splitext(os.path.basename(aoi_geometry_file))[0]
    filename = (
        start_date.replace("-", "_")
        + "_"
        + end_date.replace("-", "_")
        + "_"
        + base_geometry_name
        + ".parquet"
    )
    parq_dir = os.path.join(data_dir, "parquet")
    if not os.path.isdir(parq_dir):
        os.mkdir(parq_dir)
    return os.path.join(parq_dir, filename)


def planetary_computer_stac_query(
    start_date="2023-05-01",
    end_date="2023-08-01",
    aoi_geometry_file=os.path.join("tests", "data-test", "geojson", "sample.geojson"),
    kwargs_dict=None,
    outfile_name="Auto",
) -> os.PathLike:
    """
    Downloads a STAC catalog for a given geometry and daterange and saves
    it to a geoparquet file.

    Parameters
    ----------
    start_date : str of format YYYY-MM-DD
                start date for period of interest
    end_date : str of format YYYY-MM-DD
              end date for period of interest
    aoi_geometry_file : str | path
                        geometry to clip mosaic to, defaults to a sample
                        test geojson over Kolwezi, Southern DRC
    kwargs_dict : dict
                  pass in keyword arguments to filter your item search,
                  refer to planetary computer filter fields. eg.,
                  {"eo:cloud_cover": {"lt": 10}, "s2:nodata_pixel_percentage": {"lt":20}}
    outfile_name : str | path
                   location where to write out the catalog in parquet format.
                   'Auto' will write to the parquet subdir in the data directory.
    """
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    with open(aoi_geometry_file) as f:
        data = json.load(f)
    time_of_interest = start_date + "/" + end_date
    area_of_interest = data["features"][0]["geometry"]
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=area_of_interest,
        datetime=time_of_interest,
        query=kwargs_dict,
    )

    items = search.item_collection()
    item_list = [item.to_dict() for item in items]
    df = stac_geoparquet.to_geodataframe(item_list)
    if outfile_name == "Auto":
        outfile_name = _make_parquet_filename(start_date, end_date, aoi_geometry_file)
    df.to_parquet(outfile_name)
    return outfile_name
