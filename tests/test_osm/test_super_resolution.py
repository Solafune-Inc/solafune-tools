import numpy as np
from solafune_tools.osm.super_resolution.inference import Model
from tifffile import TiffFile
import cv2

def test_load_weights():
    # Test case: Load weights
    inference_model = Model()
    assert len(inference_model.list_model) > 0

inference_model = Model()

# Test cases for Single Input Inference
def test_single_small_pixel_image():
    # Test case 1: Small pixel image
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    output = inference_model.single_input_inference(image)
    assert output.shape == (650, 650, 3)

def test_single_black_image():
    # Test case 2: Black image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    output = inference_model.single_input_inference(image)
    assert output.shape == (650, 650, 3)

def test_single_white_image():
    # Test case 3: White image
    image = np.ones((100, 100, 3), dtype=np.uint8)
    output = inference_model.single_input_inference(image)
    assert output.shape == (650, 650, 3) 
    
def test_single_random_image():
    # Test case 4: Random image
    image = np.random.randint(0, 255, size=(30, 50, 3), dtype=np.uint8)
    output = inference_model.single_input_inference(image)
    assert output.shape == (650, 650, 3)

def test_single_ruled_image():
    # Test case 5: Large image
    image = np.ones((130, 130, 3), dtype=np.uint8)
    output = inference_model.single_input_inference(image)
    assert output.shape == (650, 650, 3)

# Test cases for Multi Input Inference
def test_multi_small_pixel_image():
    # Test case 1: Small pixel image
    images = [np.zeros((5, 5, 3), dtype=np.uint8) for i in range(5)]
    output = inference_model.multi_input_inference(images)
    assert len(output) == 5
    assert output[0].shape == (650, 650, 3)

def test_multi_black_image():
    # Test case 2: Black image
    image = [np.zeros((100, 100, 3), dtype=np.uint8) for i in range(5)]
    output = inference_model.multi_input_inference(image)
    assert len(output) == 5
    assert output[0].shape == (650, 650, 3)

def test_multi_white_image():
    # Test case 3: White image
    image = [np.ones((100, 100, 3), dtype=np.uint8)  for i in range(5)]
    output = inference_model.multi_input_inference(image)
    assert len(output) == 5
    assert output[0].shape == (650, 650, 3)
    
def test_multi_random_image():
    # Test case 4: Random image
    image = [np.random.randint(0, 255, size=(30, 50, 3), dtype=np.uint8)  for i in range(5)]
    output = inference_model.multi_input_inference(image)
    assert len(output) == 5
    assert output[0].shape == (650, 650, 3)

def test_multi_ruled_image():
    # Test case 5: Large image
    image = [np.ones((130, 130, 3), dtype=np.uint8)  for i in range(5)]
    output = inference_model.multi_input_inference(image)
    assert len(output) == 5
    assert output[0].shape == (650, 650, 3)

# Test cases for real-image generation
## Use normal input 130x130 pixel images
def test_real_image_generation():
    # Test case: Real image generation
    with TiffFile("tests/data-test/osm_sr/test_image_normal.tif") as tif:
        image = tif.asarray()
    output = inference_model.generate(image)

    assert isinstance(output, np.ndarray), f"Expected ndarray but got: {output}"
    assert output.shape == (650, 650, 3)
    
# Use large input 1024x1024 pixel images
def test_real_image_generation_large():
    # Test case: Real image generation with large input
    with TiffFile("tests/data-test/osm_sr/test_image_large.tif") as tif:
        image = tif.asarray()
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    output = inference_model.generate(image)

    assert isinstance(output, np.ndarray), f"Expected ndarray but got: {output}"
    assert output.shape == (5120, 5120, 3)