### solafune_tools: Internal Geodata Creation and Management Tools

This package contains tools to download STAC catalogs and Sentinel-2 imagery from Planetary Computer and assembling it into a cloudless mosaic. Other tools will be added in the future.

## Quickstart

Install the package using `pip` or `uv pip`, recommend using `python 3.10`:

``` 
uv pip install solafune_tools
```

Before using the library, you can set the directory where you want to store data by calling
```
solafune_tools.set_data_directory(dir_path="your_data_dir_here")
```
The above command sets the environment variable `solafune_tools_data_dir` from where all sub-modules draw their file paths. It is not set persistenly (i.e., not written to `.bashrc` or similar), so you will need to set it each time you ssh into your machine or on reboot. If you do not explicitly set this, it will default to creating/using a `data` folder within your current working directory.

A one-shot command exists to make a cloudless mosaic given a daterange and area of interest. 
Before running this function, create a Dask server and client. This function uses lazy chunked xarray Dataarrays which can (and should) be processed in parallel. The simplest way to do so is to open a Jupyter notebook and paste the following code into it. If you call this from within a python script, you need to put it under a ` if __name__ == "__main__":` block to work.

```python
from dask.distributed import Client, LocalCluster

cluster = LocalCluster()
client = Client(cluster)
client
```
It will print out a dashboard link for your cluster that you can use to track the progress of your function. The actual function call is below.


```python
mosaics_catalog = solafune_tools.create_basemap(
    start_date="2023-05-01",
    end_date="2023-08-01",
    aoi_geometry_file="data/geojson/xyz_bounds.geojson",
    bands=["B02", "B03", "B04"],
    mosaic_epsg="Auto",
    mosaic_resolution=100,
    clip_to_aoi=True,
)
```

If you want your mosaic broken up into tiles, pass in a tile_size argument (size in pixels). Tiles for the below call will be 100x100 except that the right and bottom boundaries of the mosaic where they maybe rectangular and smaller due to the mosaic not accomodating an integer number of tiles.

```python
mosaics_catalog = solafune_tools.create_basemap(
    start_date="2023-05-01",
    end_date="2023-08-01",
    aoi_geometry_file="data/geojson/xyz_bounds.geojson",
    bands=["B02", "B03", "B04"],
    mosaic_epsg="Auto",
    mosaic_resolution=100,
    clip_to_aoi=True,
    tile_size=100,
)
```
The output is a link to a STAC catalog of all mosaics generated so far in the current data directory. See point 6 in the workflow below to see how to load and query it.

A typical workflow to assemble a cloudless mosaic is as follows. I strongly recommend leaving all outfile and outdirectory naming to 'Auto' if you choose to run these functions one by one.

1. Get the Sentinel-2 catalog items for your area of interest (pass in a geojson) and a date range.
```python
plc_stac_catalog = solafune_tools.planetary_computer_stac_query(
    start_date="2023-05-01",
    end_date="2023-08-01",
    aoi_geometry_file= "data/geojson/xyz_bounds.geojson",
    outfile_name='Auto'
)
```

2. Download files for the bands you want for these catalog items.

```python
tiffile_dir = solafune_tools.planetary_computer_fetch_images(
    dataframe_path=plc_stac_catalog,
    bands=["B02", "B03", "B04"],
    outfile_dir='Auto',
)
```
3. Assemble a STAC catalog of local files (this is necessary for mosaicking)

```python
local_stac_catalog = solafune_tools.create_local_catalog_from_existing(
    input_catalog_parquet=plc_stac_catalog,
    bands=["B04", "B03", "B02"],
    tif_files_dir=tiffile_dir,
    outfile_dir='Auto',
)
```
4. Make a cloudless mosaic. Make sure to have a Dask cluster running for this step. Otherwise, it will either take days to finish or crash out with memory errors. Only pass in a geometry file if you want to your mosaic clipped to that geometry.

```python
mosaic_file_loc = solafune_tools.create_mosaic(
    local_stac_catalog=local_stac_catalog,
    aoi_geometry_file=None,
    outfile_loc='Auto',
    out_epsg='Auto',
    resolution=100,
)
```
5. Update the STAC catalog for the mosaics folder.
```python
mosaics_catalog = solafune_tools.create_local_catalog_from_scratch(
    infile_dir='Auto',
    outfile_loc='Auto'
    )
```

6. The STAC catalog contains the geometry, date range and bands for each mosaic tif stored in the directory. Now you can query the catalog by loading it as a `Geopandas.geodataframe` and filtering for various conditions. The links for each mosaic are stored under the column `assets` under the dictionary key `mosaic` followed by `href`. 
```python
geodataframe = solafune_tools.get_catalog_items_as_gdf(mosaics_catalog)
your_query = geodataframe.geometry.intersects(your_roi_geometry) & (geodataframe['datetime']=='2021-03-01')
results = geodataframe[your_query]
your_mosaic_tif_locs = [asset['mosaic']['href'] for asset in results.assets]
# merge your mosaic tifs, do windowed reads, whatever else you need

```