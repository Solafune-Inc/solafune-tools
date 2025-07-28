import os
from box import Box
from typing import List, Tuple

try:
    from config import CFG
    from model import SRModel
    from common_utils import slice_img, merge_img, resize_img, OSMResourceDownloader
    print("Running from internal modules")
except ImportError as e:
    print(f"ImportError: {e}. Trying to import from solafune_tools modules.")
    print("Running from solafune_tools modules")
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
from tqdm.auto import tqdm
import tqdm as std_tqdm
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
    MAX_INPUT_DIM = 2000
    SLICE_THRESHOLD = 360
    FINAL_SIZE = (650, 650)
    UPSCALE_FACTOR = 5

    def __init__(self) -> None:
        """
        Initializes the instance, loading models from local paths or downloading them if not found.
        """
        self.transform = A.Compose([ToTensorV2()], is_check_shapes=False)
        self.processor = AutoImageProcessor.from_pretrained(cfg.model_name)
        self.list_model = self._load_models()

    def _load_models_from_path(self, models_dir: str) -> List[SRModel]:
        """Helper function to load all model folds from a specific directory."""
        models = []
        for fold in range(cfg.folds):
            ckpt_path = os.path.join(models_dir, cfg.ckpt_stem + f"_fold{fold}.ckpt")
            if not os.path.exists(ckpt_path):
                # If any fold is missing, this path is invalid.
                return [] 
            model = SRModel.load_from_checkpoint(ckpt_path, cfg=cfg, fold=fold, weights_only=False)
            models.append(model.to(cfg.device))
        return models

    def _load_models(self, version:str = "v2") -> List[SRModel]:
        """
        Tries to load models from a series of potential local directories.
        If all attempts fail, it downloads the models and then loads them.
        """
        # Define potential local directories in order of preference
        potential_dirs = [
            os.path.join(os.getcwd(), "osm/super_resolution/weights"),
            os.path.join(os.getcwd(), "weights"),
        ]

        for models_dir in potential_dirs:
            print(f"Attempting to load models from: {models_dir}")
            loaded_models = self._load_models_from_path(models_dir)
            if loaded_models:
                print("Models loaded successfully.")
                return loaded_models

        # If models were not found in any local path, download them
        print("Local models not found. Downloading from OSM resource downloader...")
        downloader = OSMResourceDownloader()
        status, _ = downloader.model_weight_download(f"super_resolution_{version}") # Returns status, but we assume success if it doesn't raise an error
        print(f"Download status: {status}")

        models_dir = os.path.join(downloader.model_weights_dir, f"super_resolution_{version}")
        print(f"Attempting to load downloaded models from: {models_dir}")
        downloaded_models = self._load_models_from_path(models_dir)
        
        if not downloaded_models:
            # This would be a critical failure
            raise RuntimeError("Failed to load models even after downloading. Please check paths and files.")
            
        print("Downloaded models loaded successfully.")
        return downloaded_models
        
    
    def merge_output(self, grouped_outputs: List[Tuple[np.ndarray, ...]]) -> List[np.ndarray]:
        """
        Averages the predictions from multiple models for each image.
        
        Parameters:
            grouped_outputs (List[Tuple[np.ndarray, ...]]): A list where each element is a tuple 
                                                            of prediction arrays from all models for one input image.
        
        Returns:
            List[np.ndarray]: A list of final, merged images.
        """
        final_images = []
        for image_preds in grouped_outputs:
            # Stack predictions along a new axis and calculate the mean
            mean_pred = np.mean(np.array(image_preds), axis=0)
            # Round and convert to the final data type
            final_image = np.round(mean_pred, decimals=0).astype(np.uint8)
            final_images.append(final_image)
        return final_images
    
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

        # --- Conditional Logic for Progress Bar Arguments ---
        # Check if we are in a terminal environment by checking the type of tqdm.
        # tqdm.auto will select std_tqdm.tqdm for terminals.
        is_terminal = isinstance(tqdm, std_tqdm.tqdm)

        # Set arguments for the outer loop (models)
        outer_kwargs = {'desc': "Overall Model Progress", 'disable': not show_progress}
        if is_terminal:
            outer_kwargs['position'] = 0
            outer_kwargs['leave'] = True

        # Set arguments for the inner loop (batches)
        inner_kwargs = {'desc': "Batches", 'disable': not show_progress, 'leave': False}
        if is_terminal:
            inner_kwargs['position'] = 1
        # ----------------------------------------------------

        # The loops now use the dynamically set keyword arguments
        for model in tqdm(self.list_model, **outer_kwargs):
            
            batch_iter = tqdm(inference_dl, **inner_kwargs)
            
            model_outputs = []
            for batch_images in batch_iter:
                output = self.main_inference_core(model, batch_images.to(cfg.device))
                model_outputs.extend(output)
                
            output_list.append(model_outputs)
        
        # output_list is like [[model1_img1, model1_img2], [model2_img1, model2_img2], ...]
        # We want to group them as [[model1_img1, model2_img1], [model1_img2, model2_img2], ...]
        # The * operator unpacks the list for zip
        images_grouped_by_input = list(zip(*output_list))

        # Now, pass this clean structure to the merge function
        final_output = self.merge_output(images_grouped_by_input)
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
        
        if image.shape[0] > self.MAX_INPUT_DIM or image.shape[1] > self.MAX_INPUT_DIM:
            return f"Image is too large, please use an image up to {self.MAX_INPUT_DIM}x{self.MAX_INPUT_DIM} pixels"
        
        if image.shape[0] >= self.SLICE_THRESHOLD or image.shape[1] >= self.SLICE_THRESHOLD:
            print("slice")
            img_list, indices = slice_img(image)
            img_results = self.multi_input_inference([resize_img(x) for x in img_list])
            new_indices: List[Tuple[int, int, int, int]] = [(
                idx[0] * self.UPSCALE_FACTOR, 
                idx[1] * self.UPSCALE_FACTOR, 
                idx[2] * self.UPSCALE_FACTOR, 
                idx[3] * self.UPSCALE_FACTOR
            ) for idx in indices] # idx is Tuple[int, int, int, int]
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

    output_dir = os.path.dirname(img_output)
    # Ensure the output directory exists, but only if a directory path is specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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
        #img_result = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)
        tifffile.imwrite(img_output, img_result)
    else:
        cv2.imwrite(img_output, img_result)

    print(f"Processing time: {time.time() - start_time} seconds")