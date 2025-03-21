# STAC Download on Sentinel-2 Data

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
    kwargs_dict={"eo:cloud_cover": {"lt": 10}, "s2:nodata_pixel_percentage": {"lt":20}},
    bands="Auto",
    mosaic_epsg="Auto",
    mosaic_resolution=100,
    clip_to_aoi=True,
)
```

You can pass in planetary computer query fields as keyword arguments to apply them as filters on your item queries. If you want your mosaic broken up into tiles, pass in a tile_size argument (size in pixels). Tiles for the below call will be 100x100 except that the right and bottom boundaries of the mosaic where they maybe rectangular and smaller due to the mosaic not accomodating an integer number of tiles. You can also pass a list for bands like `bands = ['B02','B04']` if you want to select only certain bands to make a mosaic. Further, you can choose to make several single band mosaics or a multiband mosaic by passing in `Singleband` or `Multiband` to this function. You can also choose whether to use the `Median` or `Minimum` value of each pixel in an image stack to create the mosaic by setting `mosaic_mode`.

```python
mosaics_catalog = solafune_tools.create_basemap(
    start_date="2023-05-01",
    end_date="2023-08-01",
    aoi_geometry_file="data/geojson/xyz_bounds.geojson",
    bands=['B02','B04'],
    mosaic_epsg="Auto",
    mosaic_resolution=100,
    clip_to_aoi=True,
    tile_size=100,
    mosaic_style='Multiband',
    mosaic_mode='Median',
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
    kwargs_dict={"eo:cloud_cover": {"lt": 10}, "s2:nodata_pixel_percentage": {"lt":20}},
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
    bands='Auto',
    mosaic_style='Multiband',
    mosaic_mode='Median',
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
