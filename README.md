<h1 align="center">
<img src="https://solafune-contents.s3.ap-northeast-1.amazonaws.com/Solafune+Tools+Logo.png" width="300">
</h1><br>

### solafune_tools: Open tools for solafune's developers and solafune's hackers where can share developed tools in geospatial data

This library packages is an integrated open source tools for solafune's developers, solafune's hackers, scientist, engineer, students and anyone who interested in doing geospatial data analysis. The solafune-tools contains many tools to help you develop your analyis like for downloading STAC catalogs and Sentinel-2 imagery (from Planetary Computer) then assembing it into a cloudless mosaic, competition_tools, community_tools, and other tools in geospatial data analysis. We are welcome for your contribution in this development of geospatial tools. Other tools also will be added in the futures.

#### Oct 2024: Added a panoptic metric

## Quickstart

Install the package using `pip` or `uv pip`, recommend using `python 3.10`:

```bash
uv pip install solafune_tools
```

All public-facing functions have detailed docstrings explanining their expected inputs and outputs. You can check any of them through `print(solafune_tools.function_name.__doc__)` (if you don't use print it shows as an unstructured string) or `??solafune_tools.function_name` in jupyter notebooks.
Before using the library, you can set the directory where you want to store data by calling

```python
solafune_tools.set_data_directory(dir_path="your_data_dir_here")
```

The above command sets the environment variable `solafune_tools_data_dir` from where all sub-modules draw their file paths. It is not set persistenly (i.e., not written to `.bashrc` or similar), so you will need to set it each time you ssh into your machine or on reboot. If you do not explicitly set this, it will default to creating/using a `data` folder within your current working directory.
