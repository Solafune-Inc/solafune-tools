from PIL import Image
from PIL.Image import Image as PILImage
import PIL
from glob import glob
import random
from tqdm import tqdm
import os 
from multiprocessing import Pool

PIL.Image.MAX_IMAGE_PIXELS = 933120000

SCALE_LOW = 5
SCALE_HIGH = 1
OUTPUT_HEIGHT = 650
OUTPUT_WEIGHT = 650
TRAIN_COUNT = 0.9

file_type_list = ["tif","tiff","jpg","jpeg","png"]

def make_dataset(data_path: str, output_dir: str = "datasets"):
    
    os.makedirs(f"{output_dir}/train",exist_ok=True)
    os.makedirs(f"{output_dir}/test",exist_ok=True)
    
    exist_train = glob(f"{output_dir}/train/*.tif")
    exist_test = glob(f"{output_dir}/test/*.tif")
    piclist = []
    
    for file_type in file_type_list:
        piclist += glob(os.path.join(data_path, "**", f"*.{file_type}"), recursive=True)
        
    count = -1
    train_image_id = int((len(exist_train)/2) - 1)
    test_image_id =  int((len(exist_test)/2) - 1)
    train_count = TRAIN_COUNT * len(piclist)
    
    random.shuffle(piclist)
    
    for pic in tqdm(piclist):
        load_image: PILImage = Image.open(pic)
        count += 1
        for crop_image in image_divider(load_image):
            

            lowresoultuion_image = low_resolution(crop_image,SCALE_LOW)
            highresolution_image = low_resolution(crop_image,SCALE_HIGH)

            if(count < train_count):
                train_image_id += 1   
                lowresoultuion_image.save(f"{output_dir}/train/train_{train_image_id}_low.tif")
                highresolution_image.save(f"{output_dir}/train/train_{train_image_id}_high.tif")
                continue
            
            test_image_id += 1   
            lowresoultuion_image.save(f"{output_dir}/test/test_{test_image_id}_low.tif")
            highresolution_image.save(f"{output_dir}/test/test_{test_image_id}_high.tif")

def low_resolution(image: PILImage, scale: int):
    inputimg = image
    targetimg = inputimg.resize((inputimg.width // scale, inputimg.height // scale), Image.LANCZOS)
    return targetimg


def image_divider(inputimg: PILImage) -> Image:
    imgwidth = inputimg.width
    imgheight = inputimg.height
    for y in range(0,imgheight,OUTPUT_HEIGHT):
        for x in range(0,imgwidth,OUTPUT_WEIGHT):
            box = (x,y,x+OUTPUT_HEIGHT,y+OUTPUT_WEIGHT)
            yield inputimg.crop(box)


def train_noresize():
    for count in range(TRAIN_COUNT):
        high_image = f"dataset_competition/train/train_{count}_high.tif"
        low_image = low_resolution(high_image,4)
        low_image.save(f"dataset_competition/train_noresize/train_{count}_low.tif")

def test_noresize():
    for count in range(PUBLICLB_COUNT+PRIVATELB_COUNT):
        high_image = f"dataset_competition/test_answer/test_{count}_high.tif"
        low_image = low_resolution(high_image,4)
        low_image.save(f"dataset_competition/test_noresize/test_{count}_low.tif")

def test_noresize():
    high_image = "data/05OD9472.tif"
    highresolution_image = low_resolution(high_image,5)
    lowresolution_image = low_resolution(high_image,20)

    highresolution_image.save('testimage/highimage.tif')
    lowresolution_image.save('testimage/lowimage.tif')

def test_upresize():
    lowimage = 'testimage/lowimage.tif'
    inputimg = Image.open(lowimage)
    targetimg = inputimg.resize((inputimg.width * 4, inputimg.height * 4), Image.BICUBIC)
    targetimg.save('testimage/low_resize.tif')

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="datasets")
    args = parser.parse_args()
    make_dataset(args.data_path, args.output_dir)



