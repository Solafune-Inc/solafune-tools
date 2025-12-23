import os
import shutil
import tempfile
import pytest
import numpy as np
import rasterio
from solafune_tools.community_tools.timelapse import create_timelapse

@pytest.fixture
def temp_timelapse_data():
    """Create a temporary directory with dummy Sentinel-2 style TIFFs."""
    temp_dir = tempfile.mkdtemp()
    
    # Create 3 dummy images representing 3 dates
    filenames = [
        "T54TVL_20230101T000000_B04.tif",
        "T54TVL_20230115T000000_B04.tif",
        "T54TVL_20230201T000000_B04.tif"
    ]
    
    # Create random data
    width, height = 100, 100
    
    for i, fname in enumerate(filenames):
        array = np.zeros((height, width), dtype=np.uint16)
        start_x = i * 20
        array[30:70, start_x:start_x+20] = 5000 
        noise = np.random.randint(0, 1000, (height, width), dtype=np.uint16)
        array = array + noise
        path = os.path.join(temp_dir, fname)
        with rasterio.open(path, 'w', driver='GTiff', height=height, width=width, count=1, dtype='uint16') as dst:
            dst.write(array, 1)
            
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_create_timelapse(temp_timelapse_data):
    """Test the timelapse creation function."""
    input_dir = temp_timelapse_data
    output_gif = os.path.join(input_dir, "output.gif")
    
    result_path = create_timelapse(
        input_dir=input_dir,
        output_filename=output_gif,
        fps=2,
        add_timestamp=True,
        resize_factor=0.5
    )
    
    assert os.path.exists(output_gif)
    assert result_path == output_gif
    assert os.path.getsize(output_gif) > 0
    
    from PIL import Image
    with Image.open(output_gif) as img:
        assert img.format == "GIF"
        assert img.is_animated
        assert img.n_frames == 3

def test_create_timelapse_no_files(temp_timelapse_data):
    """Test error handling when no files are found."""
    empty_dir = os.path.join(temp_timelapse_data, "empty")
    os.mkdir(empty_dir)
    with pytest.raises(FileNotFoundError):
        create_timelapse(empty_dir, "dummy.gif")

def test_date_extraction():
    from solafune_tools.community_tools.timelapse.core import _extract_date
    assert _extract_date("T54TVL_20230523T013721_B04.tif") == "2023-05-23"
