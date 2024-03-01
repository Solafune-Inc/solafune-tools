### solafune_tools: Internal Geodata Creation and Management Tools


Before using the library, you can set the directory where you want to store data by calling
```
solafune_tools.settings.set_data_directory(dir_path="your_data_dir_here")
```
Please use absolute path not relative, this is necessary for intermediate processes that do file read and writes. The above command sets the environment variable 'solafune_tools_data_dir' from where all sub-modules draw their file paths. If you do not explicitly set this, it will default to creating a `data` folder within your current working directory.