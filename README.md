### solafune_tools: Internal Geodata Creation and Management Tools

This package contains tools to download STAC catalogs and Sentinel-2 imagery from Planetary Computer and assembling it into a cloudless mosaic. Other tools will be added in the future.

## Quickstart

Install the package using `pip`, recommend using python 3.10:

``` 
pip install solafune_tools
```

Before using the library, you can set the directory where you want to store data by calling
```
solafune_tools.set_data_directory(dir_path="your_data_dir_here")
```
The above command sets the environment variable 'solafune_tools_data_dir' from where all sub-modules draw their file paths. If you do not explicitly set this, it will default to creating a `data` folder within your current working directory.

A typical workflow to assemble a cloudless mosaic is as follows:

1. Get the Sentinel-2 catalog items for your area of interest (pass in a geojson) and a date range.
```
plc_stac_catalog = solafune_tools.planetary_computer_stac_query(
    start_date="2023-05-01",
    end_date="2023-08-01",
    aoi_geometry_file= "data/geojson/xyz_bounds.geojson"),
)
```

2. Download files for the bands you want for these catalog items.

```
solafune_tools.planetary_computer_fetch_images(
    dataframe_path=plc_stac_catalog,
    bands=["B02", "B03", "B04"],
    dest_dir=os.path.join(data_dir, "tif/"),
)
```
3. Assemble a STAC catalog of local files (this is necessary for mosaicking)

```
local_stac = solafune_tools.make_catalog(
    input_filename=plc_stac_catalog,
    bands=["B04", "B03", "B02"],
    tif_files_dir=os.path.join(data_dir, "tif/"),
    outdir=os.path.join(data_dir, "stac/"),
)
```
4. Make a cloudless mosaic

```
mosaic_file_loc = solafune_tools.make_mosaic(
    stac_catalog = local_stac,
    outfile_loc = os.path.join(
    data_dir, 
    "mosaics", 
    "outval.tif"
    ),
    epsg = None,
    resolution = 100,
):
```
