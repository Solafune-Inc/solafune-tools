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

logging.basicConfig(level=logging.INFO)
PIL.Image.MAX_IMAGE_PIXELS = 2e8
data_dir = os.getenv("solafune_tools_data_dir", "data/")


def log(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"{func.__name__} {result} time {end_time-start_time:.2f}s")
        return result

    return wrapper


def check_downloaded_file(file_path, raw_file_name) -> bool:
    try:
        Image.open(file_path)
    except Exception as e:
        with open(os.path.join(data_dir, "logs/bad_responses.txt"), "a") as logfile:
            logfile.write(f"{os.path.basename(file_path)} {raw_file_name} {e}\n")
        return False
    return True


@log
def download_tiff(raw_file_name, band, dest_dir, session) -> int:
    dest_file_loc = f"{dest_dir}/{band}/{os.path.basename(raw_file_name)}"
    if os.path.isfile(dest_file_loc) and check_downloaded_file(dest_file_loc):
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
    check_downloaded_file(dest_file_loc, raw_file_name)
    print(dest_file_loc)
    return response.status_code


def setup_directories(dest_dir, bands) -> None:
    for band in bands:
        band_dir = os.path.join(dest_dir, band)
        if not os.path.isdir(band_dir):
            os.mkdir(band_dir)
    return None


def filter_redundant_items(dataframe) -> gpd.GeoDataFrame:
    reduce_df = (
        dataframe.sort_values(by=["s2:nodata_pixel_percentage", "eo:cloud_cover"])
        .groupby(["geometry"])
        .head(5)
    )
    return reduce_df


def fetch_images(
    dataframe_path=os.path.join(data_dir, "parquet/2023_May_July_CuCoBounds.parquet"),
    bands=["B02", "B03", "B04"],
    dest_dir=os.path.join(data_dir, "tif/"),
) -> None:
    gdf = filter_redundant_items(gpd.read_parquet(dataframe_path))
    assets = gdf.assets

    _ = setup_directories(dest_dir, bands)
    total = len(gdf.assets)
    session = requests.Session()
    st = time.time()
    for count, each in enumerate(assets):
        print(f"On file {count+1} of {total}")
        for band in bands:
            print(band)
            get_file = each[band]["href"]
            download_tiff(
                raw_file_name=get_file.strip(),
                band=band,
                dest_dir=dest_dir,
                session=session,
            )
    et = time.time()
    print(et - st, "final time")
    return None


def make_parquet_filename(
    prefix, start_date, end_date, aoi_geometry_file
) -> os.PathLike:
    base_geometry_name = os.path.splitext(os.path.basename(aoi_geometry_file))[0]
    filename = (
        prefix
        + start_date.replace("-", "_")
        + "_"
        + end_date.replace("-", "_")
        + "_"
        + base_geometry_name
    )
    parq_dir = os.path.join(data_dir, "parquet")
    if not os.path.isdir(parq_dir):
        os.mkdir(parq_dir)
    return os.path.join(parq_dir, filename)


def make_stac_query(
    start_date="2023-05-01",
    end_date="2023-08-01",
    aoi_geometry_file=os.path.join(data_dir, "geojson/cu_co_prospect_bounds.geojson"),
    prefix="",
) -> os.PathLike:
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
        # query={"eo:cloud_cover": {"lt": 10}, "s2:nodata_pixel_percentage": {"lt":20}},
    )

    items = search.item_collection()
    item_list = [item.to_dict() for item in items]
    df = stac_geoparquet.to_geodataframe(item_list)
    outfile_name = make_parquet_filename(
        prefix, start_date, end_date, aoi_geometry_file
    )
    df.to_parquet(outfile_name)
    return outfile_name
