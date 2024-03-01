import geopandas as gpd
import os
import stac_geoparquet
import pystac


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
    input_filename="data/parquet/2023_May_July_CuCoBounds.parquet",
    bands=["B04", "B03", "B02"],
    tif_files_dir="data/tif/sentinel-2/",
    outdir="home/pushkar/solafune-tools/data/stac/",
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
