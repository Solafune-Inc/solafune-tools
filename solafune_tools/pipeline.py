import os

import solafune_tools

data_dir = solafune_tools.settings.get_data_directory()


def create_basemap(
    start_date="2023-05-01",
    end_date="2023-08-01",
    aoi_geometry_file=os.path.join("tests", "data-test", "geojson", "sample.geojson"),
    kwargs_dict=None,
    bands="Auto",
    mosaic_epsg="Auto",
    mosaic_resolution=100,
    tile_size=None,
    clip_to_aoi=True,
    mosaic_style="Multiband",
    mosaic_mode="Median",
) -> os.PathLike:
    """
    Creates a basemap given a geometry file, date range, bands and output resolution

    Parameters
    ----------
    start_date : str of format YYYY-MM-DD
                start date for period of interest
    end_date : str of format YYYY-MM-DD
              end date for period of interest
    aoi_geometry_file : str | path
                        geometry to clip mosaic to, defaults to a sample
                        test geojson over Kolwezi, Southern DRC
    kwargs_dict : Dict
                Keyword arguments for search filtering on planetary computer,
                such as cloud cover or no data.
    bands : str | list(str)
        Pass in a list of bands for which you need mosaics. If Auto, all bands
        are used.
    mosaic_epsg : int
              crs value for the output mosaic. 'Auto' defaults to the most
              common epsg value among the input files.
    mosaic_resolution : resolution per pixel for the output data. This is in units
                        of the crs. For Sentinel-2 data, this is meters. If you use
                        epsg:4326, it will be degrees so be careful.
    tile_size : int
                Size of square tiles in pixels for mosiac output. If none, a
                single large tif file is written out.

    mosaic_style : 'Singleband' | 'Multiband''
                    Whether a single multiband mosaic is needed or individual
                    mosaics for every band

    mosaic_mode : 'Median' | 'Minimum'
                   Which function to use to generate a mosaic pixel from the image 
                   stack
    """
    plc_stac_catalog = solafune_tools.image_fetcher.planetary_computer_stac_query(
        start_date=start_date,
        end_date=end_date,
        aoi_geometry_file=aoi_geometry_file,
        kwargs_dict=kwargs_dict,
    )

    if bands == 'Auto':
        bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11', 'B12']
        
    tiffile_dir = solafune_tools.image_fetcher.planetary_computer_fetch_images(
        dataframe_path=plc_stac_catalog,
        bands=bands,
        outfile_dir="Auto",
    )

    local_stac_catalog = solafune_tools.make_catalog.create_local_catalog_from_existing(
        input_catalog_parquet=plc_stac_catalog,
        bands=bands,
        tif_files_dir=tiffile_dir,
        outfile_dir="Auto",
    )

    clip_aoi_file = aoi_geometry_file if clip_to_aoi is True else None

    _ = solafune_tools.make_mosaic.create_mosaic(
        local_stac_catalog=local_stac_catalog,
        aoi_geometry_file=clip_aoi_file,
        outfile_loc="Auto",
        out_epsg=mosaic_epsg,
        resolution=mosaic_resolution,
        tile_size=tile_size,
        bands=bands,
        mosaic_style=mosaic_style,
        mosaic_mode=mosaic_mode,
    )

    mosaics_catalog = solafune_tools.make_catalog.create_local_catalog_from_scratch(
        infile_dir="Auto", outfile_loc="Auto"
    )

    return mosaics_catalog
