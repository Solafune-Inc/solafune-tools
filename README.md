<h1 align="center">
<img src="https://solafune-contents.s3.ap-northeast-1.amazonaws.com/Solafune+Tools+Logo.png" width="300">
</h1><br>

### solafune_tools: Internal Geodata Creation and Management Tools

This package contains tools to download STAC catalogs and Sentinel-2 imagery from Planetary Computer and assembling it into a cloudless mosaic. Other tools will be added in the future.

#### Oct 2024: Added a panoptic metric

## Quickstart

Install the package using `pip` or `uv pip`, recommend using `python 3.10`:

``` 
uv pip install solafune_tools
```
All public-facing functions have detailed docstrings explanining their expected inputs and outputs. You can check any of them through `print(solafune_tools.function_name.__doc__)` (if you don't use print it shows as an unstructured string) or `??solafune_tools.function_name` in jupyter notebooks.
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

## Metrics Toolset
In this repository we also add metrics toolset to accomodate user to be able use metrics we use usually for model training or model evaluation.

### Panoptic Qualities Score Compute Function
This function computes the Panoptic Qualities (PQ) score for a given set of predictions and ground truth. The PQ score is a metric used to evaluate the quality of panoptic segmentation models. The PQ score is computed as the sum of the PQ scores for each class in the dataset. The PQ score for a class is computed as the sum of the true positive, false positive, and false negative values for that class. The PQ score is then normalized by the sum of the true positive and false negative values for that class. The PQ score is a value between 0 and 1, with 1 being the best possible score.

https://arxiv.org/abs/1801.00868

### Authors information
Author: Toru Mitsutake(Solafune) \
Solafune Username: sitoa

### Gettin Started with PQ score
```python
from shapely.geometry import Polygon
from solafune_tools.metrics import PanopticMetric
PQ = PanopticMetric()

polygon1 = Polygon([(1, 2), (2, 4), (3, 1)])
polygon2 = Polygon([(0, 0), (1, 3), (2, 2), (3, 0)])
polygon3 = Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3)])
polygon4 = Polygon([(2, 2), (3, 4), (4, 4), (5, 2), (3, 1)])

ground_truth_polygons = [polygon1, polygon2]
prediction_polygons = [polygon3, polygon4]

pq, sq, rq = PQ.compute_pq(ground_truth_polygons, prediction_polygons)

print("PQ: ", pq)
print("SQ: ", sq)
print("RQ: ", rq)
```

### Input
- ground_truth_polygons: List of polygons representing the ground truth segmentation.
- prediction_polygons: List of polygons representing the predicted segmentation.

polygons is shapely.geometry.Polygon object
https://shapely.readthedocs.io/en/stable/

### Output
- pq: Panoptic Quality score
- sq: Segmentation Quality score
- rq: Recognition Quality score
