import os

import geopandas as gpd
import pystac
import stac_geoparquet
from settings import set_data_directory

if os.getenv("solafune_tools_data_dir") is None:
    set_data_directory()
data_dir = os.getenv("solafune_tools_data_dir")


def get_local_filename(tif_dir, band, remote_href):
    cwd = os.getcwd()
    basename = os.path.basename(remote_href)
    outfile_loc = os.path.join(cwd, tif_dir, band, basename)
    return outfile_loc


def local_file_links_update(row, bands, tif_dir):
    new_row = {band: row[band] for band in bands}
    for band in new_row.keys():
        new_row[band]["href"] = get_local_filename(tif_dir, band, row[band]["href"])
    return new_row


def create_local_catalog(
    input_filename=os.path.join(data_dir, "parquet/2023_May_July_CuCoBounds.parquet"),
    bands=["B04", "B03", "B02"],
    tif_files_dir=os.path.join(data_dir, "tif/sentinel-2/"),
    outdir=os.path.join(data_dir, "stac/"),
):
    parq = gpd.read_parquet(input_filename)
    parq.assets = parq.assets.apply(
        local_file_links_update, bands=bands, tif_dir=tif_files_dir
    )
    catalog = pystac.Catalog(
        id="test-2023", description="This catalog shows scenes over Mutanda in 2021."
    )
    item_collection = stac_geoparquet.to_item_collection(parq)
    for item in item_collection:
        catalog.add_item(item)
    catalog.normalize_hrefs(outdir)
    try:
        catalog.validate_all()
    except pystac.STACValidationError as e:
        print(e)
    catalog.save(catalog_type=pystac.CatalogType.ABSOLUTE_PUBLISHED)
    return os.path.join(outdir, "catalog.json")
