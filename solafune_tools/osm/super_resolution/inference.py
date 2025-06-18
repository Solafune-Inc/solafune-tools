import os
from box import Box
from typing import List, Tuple

try:
    print("Running from solution modules")
    from config import CFG
    from model import SRModel
    from common_utils import slice_img, merge_img, resize_img, OSMResourceDownloader
except:
    from solafune_tools.osm.super_resolution.config import CFG
    from solafune_tools.osm.super_resolution.model import SRModel
    from solafune_tools.osm.super_resolution.common_utils import slice_img, merge_img, resize_img, OSMResourceDownloader

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import AutoImageProcessor
import cv2
import argparse
import tifffile
import time
from tqdm import tqdm
from typing import Union

# Config

cfg = Box({k: v for k, v in dict(vars(CFG(debug=False, datasets_dir="",
                                              batch_size=8, grad_clip=100.0, # Batch_size for val_loader is calculated using 2*batch_size so if batch size is 1 then the val batch size is 2 and if 2 then 4
                                              learning_rate=3e-5, augmentation_end_epoch=27,
                                              cutmix_end_epoch=22, extra_dataset=True,
                                              extra_dataset_end_epoch=17, train_loader_workers=4,
                                              val_loader_workers=4, awp_start_epoch=35,
                                              batch_size_log=True, use_total_steps=True))).items() if "__" not in k})

class TeamNInferenceDataset(Dataset):
    def __init__(self, image_list):
        """
        Initializes the object with the given image list.

        Parameters:
            image_list (list): A list of images.

        Returns:
            None
        """
        self.image_list = image_list
        self.transform = A.Compose([ToTensorV2()], is_check_shapes=False)
        self.processor = AutoImageProcessor.from_pretrained(cfg.model_name)

    def __len__(self):
        """
        Returns the length of the image list.

        :return: An integer representing the number of images in the list.
        """
        return len(self.image_list)
    
    def __getitem__(self, idx):
        """
        Get an item from the image list by index.

        Parameters:
            idx (int): The index of the item to retrieve.

        Returns:
            image (Tensor): The processed image.
        """
        image = self.image_list[idx]
        image = self.transform(image=image)["image"]
        image = self.processor(image, return_tensors="pt")["pixel_values"].squeeze(0)
        return image

class Model:
    def __init__(self) -> None:
        """
        Initializes an instance of the class consisting models want to use.

        Parameters:
            None.

        Returns:
            None.
        """
        self.transform = A.Compose([ToTensorV2()], is_check_shapes=False)
        self.processor = AutoImageProcessor.from_pretrained(cfg.model_name)
        try:
            # Use trained models from recent training
            try:
                models_dir = os.path.join(os.getcwd(), "osm/super_resolution/weights")
                self.list_model = []
                for fold in range(cfg.folds):
                    model = SRModel.load_from_checkpoint(os.path.join(models_dir, (cfg.ckpt_stem + f"_fold{fold}.ckpt")), cfg=cfg, fold=fold)
                    self.list_model.append(model.to(cfg.device))
            except:
                models_dir = os.path.join(os.getcwd(), "weights")
                self.list_model = []
                for fold in range(cfg.folds):
                    model = SRModel.load_from_checkpoint(os.path.join(models_dir, (cfg.ckpt_stem + f"_fold{fold}.ckpt")), cfg=cfg, fold=fold)
                    self.list_model.append(model.to(cfg.device))
        except:
            # If the models are not found, download them from the OSM resource downloader
            print("Models not found, downloading from OSM resource downloader")
            downloader = OSMResourceDownloader()
            models_dir = downloader.model_weights_dir
            models_dir = os.path.join(models_dir, "super_resolution")

            # Download the model weights
            downloader.model_weight_download("super_resolution")
            self.list_model = []
            for fold in range(cfg.folds):
                model = SRModel.load_from_checkpoint(os.path.join(models_dir, (cfg.ckpt_stem + f"_fold{fold}.ckpt")), cfg=cfg, fold=fold)
                self.list_model.append(model.to(cfg.device))
        
    
    def merge_output(self, output_list):
        """
        Merge the given list of outputs into a single output array.
        
        Parameters:
            output_list (list): A list of output arrays to be merged.
        
        Returns:
            np.ndarray: The merged output array.
        """
        if not isinstance(output_list[0], np.ndarray):
            final_output = [np.round(np.mean(np.array(image), axis=0), decimals=0).astype(np.uint8) for image in output_list]
        else:
            output_list = np.array(output_list)
            output_list = np.mean(output_list, axis=0)
            final_output = np.round(output_list, decimals=0).astype(np.uint8)
        return final_output
    
    @torch.inference_mode()
    def main_inference_core(self, model, images):
        output = model(images)
        output = torch.round(output, decimals=0)
        output = output.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
        return output

    def multi_input_inference(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Perform inference on the given list of image.

        Parameters:
            list of image (List[np.ndarray]): The input image.

        Returns:
            List[np.ndarray]: The final output image.

        Notes:
            - This function assumes that the input image has already been preprocessed.
        """
        inference_ds = TeamNInferenceDataset(images)
        inference_dl = DataLoader(inference_ds, batch_size=cfg.val_loader.batch_size, shuffle=False)
        
        output_list = []
        show_progress = getattr(cfg, "show_progress", True)
        data_iter = tqdm(self.list_model, desc="Models", disable=not show_progress)
        for model in data_iter:
            batch_iter = tqdm(inference_dl, desc="Batches", leave=False, disable=not show_progress)
            output = [self.main_inference_core(model, batch_images.to(cfg.device)) for batch_images in batch_iter]
            output_list.append([image for batches_output in output for image in batches_output])
        
        images = []
        for i, output in enumerate(output_list):
            for j, image in enumerate(output):
                if i == 0:
                    images.append([image]) # type: ignore # Initialize the list of images with the first model's output
                else:
                    images[j].append(image) # type: ignore # Append the output of the subsequent models to the corresponding image

        final_output = self.merge_output(images)
        final_output = [cv2.resize(image, dsize=(650, 650), interpolation = cv2.INTER_CUBIC) for image in final_output]

        return final_output
    
    def single_input_inference(self, image: np.ndarray):
        """
        Perform inference on the given image.

        Parameters:
            image (np.ndarray): The input image.

        Returns:
            np.ndarray: The final output image.

        Notes:
            - This function assumes that the input image has already been preprocessed.
        """
        output_list = []
        transformed_image = self.transform(image=image)["image"]
        transformed_image = self.processor(transformed_image, return_tensors="pt")["pixel_values"]
        image = transformed_image.to(cfg.device)
        for model in self.list_model:
            output = self.main_inference_core(model, image)
            output_list.append(output)
        final_output = self.merge_output(output_list)[0]
        final_output = cv2.resize(final_output, dsize=(650, 650), interpolation = cv2.INTER_CUBIC)

        return final_output
    
    def generate(self, image: np.ndarray) -> Union[str, np.ndarray]:
        """
        Generates a 5x super resolution image from the input image using either single or multi-input inference.
        Args:
            image (np.ndarray): The input image as a NumPy array.
        Returns:
            np.ndarray or str: The super-resolved image as a NumPy array of type uint8, or an error message string if the input is invalid.
        Notes:
            - If the input is not a NumPy array, returns an error message.
            - If the image dimensions exceed 2000x2000 pixels, returns an error message.
            - For images with height or width >= 360 pixels, the image is sliced, processed in parts, and then merged.
            - For smaller images, single input inference is used.
        """
        
        if type(image) != np.ndarray:
            return "Image is empty or this is not an image as expected"
        
        if image.shape[0] > 2000 or image.shape[1] > 2000:
            return "Image is too large, please enter an image with a maximum size of 2000x2000 pixels"
        
        if image.shape[0] >= 360 or image.shape[1] >= 360:
            print("slice")
            img_list, indices = slice_img(image)
            img_results = self.multi_input_inference([resize_img(x) for x in img_list])
            new_indices: List[Tuple[int, int, int, int]] = [(idx[0] * 5, idx[1] * 5, idx[2] * 5, idx[3] * 5) for idx in indices] # idx is Tuple[int, int, int, int]
            img_result = merge_img(img_results, new_indices).astype(np.uint8)
            
        else:
            img_result = self.single_input_inference(resize_img(image))

        return img_result.astype(np.uint8)

if __name__ == "__main__":
    start_time = time.time()
    parser = argparse.ArgumentParser(prog="Super Resolution Inference API for Team Ns' Solution",
                                     description="Inference input image 150x150 and upscale the the image into 650x650")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="output/output.tif")
    args = parser.parse_args()
    img_input = str(args.input)
    img_output = str(args.output)

    if img_input.endswith(".tif") or img_input.endswith(".tiff"):
        img_input = tifffile.TiffFile(img_input).asarray() # Make sure the input is in RGB bands
    else:
        img_input = cv2.imread(img_input)

    swin2sr_teamN_infer = Model()

    img_result = swin2sr_teamN_infer.generate(img_input)
    if isinstance(img_result, str):
        print(img_result)
        exit(1)

    if img_output == "output/output.tif":
        os.makedirs(img_output.split("/")[0], exist_ok=True)

    if img_output.endswith(".tif") or img_output.endswith(".tiff"):
        img_result = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)
        tifffile.imwrite(img_output, img_result)
    else:
        cv2.imwrite(img_output, img_result)

    print(f"Processing time: {time.time() - start_time} seconds")