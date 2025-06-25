import math
import time
from pathlib import Path
import pandas as pd
import tifffile
from natsort import natsorted
import numpy as np
from tqdm import tqdm
import shutil
import copy
from box import Box
from pytorch_lightning import LightningDataModule
from empatches import EMPatches
from typing import Tuple, Union
import cv2
import os
import requests
import zipfile
from bs4 import BeautifulSoup
from typing import List, Tuple, Dict, Any

# Initialize EMPatches for image patch extraction and merging
img_patches = EMPatches()

class OSMResourceDownloader:
    """
    A class for handling model weights and dataset-related operations, including model weights, dataset downloading, and directory management.
    This class provides functionality to download specific model weights and dataset from a given URL, unzip it, and manage the directory structure of the dataset for further processing.

    Methods:
        - __init__(self) -> None: Initializes the Datasets class and creates necessary directories.
    """
    def __init__(self, model_name = None) -> None:
        """
        Initializes the OSMResourceDownloader class and sets up the necessary directories and URLs for datasets.
        Parameters:
            - model_name (str, optional): The name of the model for which weights are to be downloaded. Defaults to None.
        Returns:
            - None: This method does not return any value, but it sets up the instance with the necessary attributes.
        
        """
        
        self.dataset_url = ["https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/supoerresolution_2022/train.zip",
                            "https://solafune-dev-v1.s3.us-west-2.amazonaws.com/competitions/supoerresolution_2022/test.zip"]
        self.datasets_dir = os.path.join(os.getcwd(), "datasets/")
        self.chunk_size = 1024
        os.makedirs(self.datasets_dir, exist_ok=True)

        self.model_url = {
            "super_resolution": "https://ywbul658zi.execute-api.us-west-2.amazonaws.com/dev/download/5x_super_resolution"
        }
        user_dir = os.path.expanduser(os.path.join("~", "temp"))
        self.model_weights_dir = os.path.join(user_dir, "weights")
        os.makedirs(self.model_weights_dir, exist_ok=True)

    def model_weight_download(self, model_name: str, redownload=False) -> Union[str, Tuple[str, str]]:
        """
        Download model weights from a specified URL and save them to a local directory.

        Parameters:
            - model_name (str): The name of the model for which weights are to be downloaded.

        Returns:
            - str: A message indicating the status of the download.
            - Tuple[str, str]: A tuple containing the status message and the path to the downloaded model weights file.
        
        Raises:
            - ValueError: If the specified model name is not found in the predefined model names.
        """
        if model_name not in self.model_url:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.model_url.keys())}")

        weights_dir = os.path.join(self.model_weights_dir, model_name)
        os.makedirs(weights_dir, exist_ok=True)
        url = self.model_url[model_name]
        filename = os.path.join(self.model_weights_dir, url.split("/")[-1])+ ".zip"

        if os.path.exists(filename):
            if not redownload:
                return "Model weights already exist, redownload disabled", filename
            else:
                os.remove(filename)
        response = requests.get(url, stream=True)

        if response.status_code != 200:
            return "Error downloading, contact the author of the Solafute-Tools", filename
        
        # Save the downloaded file locally
        with open(filename, 'wb') as f:
            pbar = tqdm(desc=f"Downloading model weights...{filename}", unit="B", total=int(response.headers['Content-Length']))
            for chunk in response.iter_content(chunk_size=self.chunk_size): 
                if chunk: # filter out keep-alive new chunks
                    pbar.update(len(chunk))
                    f.write(chunk)
            pbar.close()
            f.close()
        # Unzip the downloaded file
        with zipfile.ZipFile(filename, 'r') as zip_file:
            pbar = tqdm(desc=f"Unzipping file...{filename}", iterable=zip_file.namelist(), total=len(zip_file.namelist()))
            for file in pbar:
                zip_file.extract(member=file, path=weights_dir)
            pbar.close()
            zip_file.close()
        
        return f"Model weights downloaded and unzipped to folder {weights_dir}", filename
        

    def base_dataset_download(self, redownload:bool = False) -> Tuple[str, List[str]]:
        """
        Download and unzip datasets.
        Downloads a zip file containing datasets from a URL, unzips it, and saves the files locally.
        Parameters:
            - redownload (bool): If True, re-downloads the datasets even if they already exist locally.

        Returns:
            Tuple containing:
                - Status message (str): A message indicating the outcome of the download and unzip process.
                - Dataset directory path (List[str]): The path to the directory where datasets are saved. None if download failed.

        Notes:
            - If the 'redownload' flag is set to False and the datasets already exist locally, they won't be re-downloaded.
            - This function uses the 'tqdm' library to display download and unzip progress.

        Examples:
            # Download and unzip datasets with redownload enabled
            status, dataset_dirs = download_datasets(redownload=True)

            # Download and unzip datasets without redownloading if they already exist locally
            status, dataset_dirs = download_datasets(redownload=False)
        """
        datasets_dir_list = []
        for index, url in enumerate(self.dataset_url):
            zip_filename = os.path.join(self.datasets_dir, url.split("/")[-1])
            dataset_file_dir = os.path.join(self.datasets_dir, url.split("/")[-1].split(".")[0])
            datasets_dir_list.append(dataset_file_dir)

            if os.path.exists(zip_filename):
                if not redownload:
                    if index == len(self.dataset_url) - 1:
                        return "All datasets exists, redownload disable, use existing datasets", datasets_dir_list
                    continue
                    
            response = requests.get(url, stream=True)
            if response.status_code != 200:
                return "Error downloading, check your URL", datasets_dir_list

            # Save the downloaded zip file locally
            with open(zip_filename, 'wb') as f:
                pbar = tqdm(desc=f"Downloading dataset...{zip_filename}", unit="B", total=int(response.headers['Content-Length']))
                for chunk in response.iter_content(chunk_size=self.chunk_size): 
                    if chunk: # filter out keep-alive new chunks
                        pbar.update(len(chunk))
                        f.write(chunk)
                pbar.close()
                f.close()

            # Unzip the downloaded file
            with zipfile.ZipFile(zip_filename, 'r') as zip_file:
                pbar = tqdm(desc=f"Unzipping file...{zip_filename}", iterable=zip_file.namelist(), total=len(zip_file.namelist()))
                for file in pbar:
                    zip_file.extract(member=file, path=dataset_file_dir)
                pbar.close()
                zip_file.close()

        return f"Datasets downloaded and unzipped to folder {datasets_dir_list}", datasets_dir_list
    
    def download_extra_datasets(self):
        """
        Downloads extra datasets from a given URL and saves them locally.

        Returns:
            str: A message indicating the status of the download and unzipping process.
        """
        
        if os.path.exists(os.path.join(self.datasets_dir, "nerima_dataset")):
            return "Extra datasets already downloaded and unzipped"
        
        load_url = "https://catalog.data.metro.tokyo.lg.jp/dataset/t131202d0000000108"
        extras_dir = "extra_datasets"
        os.makedirs(os.path.join(self.datasets_dir, extras_dir), exist_ok=True)
        html = requests.get(load_url)
        soup = BeautifulSoup(html.content, "html.parser")

        # HTML全体を表示する
        listurl = soup.select('a[href$="zip"]')
        for link in listurl:
            filepath = link.get('href')
            zip_filename = os.path.join(self.datasets_dir, extras_dir, os.path.basename(filepath)) # type: ignore  
            response = requests.get(url=str(filepath), stream=True)
            if response.status_code != 200:
                return "Error downloading, check your URL or connection"
            
            # Save the downloaded zip file locally
            with open(zip_filename, 'wb') as f:
                pbar = tqdm(desc=f"Downloading dataset...{zip_filename}", unit="B", total=int(response.headers['Content-Length']))
                for chunk in response.iter_content(chunk_size=self.chunk_size): 
                    if chunk: # filter out keep-alive new chunks
                        pbar.update(len(chunk))
                        f.write(chunk)
                pbar.close()
                f.close()

            # Unzip the downloaded file
            with zipfile.ZipFile(zip_filename, 'r') as zip_file:
                pbar = tqdm(desc=f"Unzipping file...{zip_filename}", iterable=zip_file.namelist(), total=len(zip_file.namelist()))
                for file in pbar:
                    zip_file.extract(member=file, path=os.path.join(self.datasets_dir, extras_dir))
                pbar.close()
                zip_file.close()

        OUTPUT_PATH = os.path.join(self.datasets_dir, "nerima_dataset")
        os.makedirs(OUTPUT_PATH, exist_ok=True) 
        i = 0
        img_size = 650 * 3

        for root, dirs, files in tqdm(desc="Creating dataset and transforming images from high to low resolution", iterable=os.walk(os.path.join(self.datasets_dir, extras_dir))):
            for fname in files:
                filepath = os.path.join(root, fname)
                if filepath.endswith(".tif") == False:
                    continue
                img = tifffile.TiffFile(f"{filepath}").asarray()

                r_chunk = img.shape[1] // img_size
                c_chunk = img.shape[2] // img_size

                for r in range(r_chunk):
                    for c in range(c_chunk):
                        # print(img.shape)
                        high_chunk = img[
                            :, r * img_size : (r + 1) * img_size, c * img_size : (c + 1) * img_size
                        ].transpose(1, 2, 0)

                        # print(high_chunk.shape)
                        high_chunk = cv2.resize(
                            high_chunk, dsize=(650, 650), interpolation=cv2.INTER_CUBIC
                        )
                        low_chunk = cv2.resize(
                            high_chunk, (130, 130), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
                        )
                        cv2.imwrite(f"{OUTPUT_PATH}/train_{i}_high.tif", high_chunk)
                        cv2.imwrite(f"{OUTPUT_PATH}/train_{i}_low.tif", low_chunk)
                        i = i + 1
        shutil.rmtree(os.path.join(self.datasets_dir, extras_dir))
        return f"Extra datasets downloaded and unzipped to folder {OUTPUT_PATH}"

def get_total_n_steps(datamodule: LightningDataModule, epochs: int) -> int:
    """
    Calculate the total number of steps based on the provided LightningDataModule and number of epochs.

    Args:
        datamodule (LightningDataModule): The LightningDataModule object containing the data loaders.
        epochs (int): The number of epochs.

    Returns:
        int: The total number of steps.
    """
    datamodule = copy.deepcopy(datamodule)
    ret = 0
    for i in range(epochs):
        datamodule.trainer = Box({"current_epoch": i}) # type: ignore
        ret += len(datamodule.train_dataloader())
    return ret

def delete_if_file_exists(path: Path) -> bool:
    """
    Delete the file at the given path if it exists.

    Parameters:
        path (Path): The path to the file.

    Returns:
        bool: True if the file was deleted, False if the file does not exist.
    """
    if path.exists():
        path.unlink()
        return True
    return False


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        """
        Initializes the object.

        This function is the constructor for the class. It initializes the object by calling the `reset` method.

        Parameters:
            self: The object itself.

        Returns:
            None
        """
        self.reset()

    def reset(self):
        """
        Reset the state of the object.

        This function sets the values of the 'val', 'avg', 'sum', and 'count' attributes to 0.

        Parameters:
            self (object): The object instance.

        Returns:
            None
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the state of the object by adding a new value.

        Parameters:
            val (int): The value to be added.
            n (int, optional): The number of times the value should be added. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s: int | float) -> str:
    """
    Convert the given time duration in seconds to minutes and seconds.

    Parameters:
        s (int | float): The time duration in seconds.

    Returns:
        str: The time duration formatted as minutes and seconds.
    """
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since: int | float, percent: float) -> str:
    """
    Calculates the time elapsed since a given starting point and the estimated remaining time based on a percentage.

    Parameters:
        since (int | float): The starting point in seconds.
        percent (float): The completion percentage as a decimal value between 0 and 1.

    Returns:
        str: A string representation of the elapsed time and the estimated remaining time in the format "elapsed_time (remain remaining_time)".
    """
    # assert 0 <= percent <= 1
    if percent == 0:
        return "0m 0s (remain ?m ?s)"
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (remain %s)" % (asMinutes(s), asMinutes(rs))


class RemainedTimeEstimator:
    def __init__(self):
        """
        Initializes the object.

        Parameters:
            None

        Returns:
            None
        """
        self.start = time.time()
        self.last = self.start

    def __call__(self, percent: float) -> str:
        """
        Calculate the time elapsed since the start of the process as a percentage of the total time.

        Args:
            percent (float): The percentage of the total time elapsed.

        Returns:
            str: The time elapsed since the start of the process as a formatted string.
        """
        return timeSince(self.start, percent)

def load_image_info(row: pd.Series):
    """
    Load image information from a pandas Series object.

    Parameters:
        row (pd.Series): A pandas Series object containing the image information.

    Returns:
        tuple: A tuple containing the following image information:
            - h_H (int): The height of the high resolution image.
            - w_H (int): The width of the high resolution image.
            - dtype_h (dtype): The data type of the high resolution image.
            - img_h[:, :, 0].mean() (float): The mean value of the red channel of the high resolution image.
            - img_h[:, :, 0].std() (float): The standard deviation of the red channel of the high resolution image.
            - img_h[:, :, 1].mean() (float): The mean value of the green channel of the high resolution image.
            - img_h[:, :, 1].std() (float): The standard deviation of the green channel of the high resolution image.
            - img_h[:, :, 2].mean() (float): The mean value of the blue channel of the high resolution image.
            - img_h[:, :, 2].std() (float): The standard deviation of the blue channel of the high resolution image.
            - h_L (int): The height of the low resolution image.
            - w_L (int): The width of the low resolution image.
            - dtype_l (dtype): The data type of the low resolution image.
            - img_l[:, :, 0].mean() (float): The mean value of the red channel of the low resolution image.
            - img_l[:, :, 0].std() (float): The standard deviation of the red channel of the low resolution image.
            - img_l[:, :, 1].mean() (float): The mean value of the green channel of the low resolution image.
            - img_l[:, :, 1].std() (float): The standard deviation of the green channel of the low resolution image.
            - img_l[:, :, 2].mean() (float): The mean value of the blue channel of the low resolution image.
            - img_l[:, :, 2].std() (float): The standard deviation of the blue channel of the low resolution image.
    """
    P_H = row.path_high
    img_h = tifffile.imread(P_H)
    h_H, w_H, _ = img_h.shape
    dtype_h = img_h.dtype

    P_L = row.path_low
    img_l = tifffile.imread(P_L)
    h_L, w_L, _ = img_l.shape
    dtype_l = img_l.dtype

    return (h_H, w_H, dtype_h,
            img_h[:, :, 0].mean(), # type: ignore
            img_h[:, :, 0].std(), # type: ignore
            img_h[:, :, 1].mean(), # type: ignore
            img_h[:, :, 1].std(), # type: ignore
            img_h[:, :, 2].mean(), # type: ignore
            img_h[:, :, 2].std(), # type: ignore
            h_L,
            w_L,
            dtype_l,
            img_l[:, :, 0].mean(), # type: ignore
            img_l[:, :, 0].std(), # type: ignore
            img_l[:, :, 1].mean(), # type: ignore
            img_l[:, :, 1].std(), # type: ignore
            img_l[:, :, 2].mean(), # type: ignore
            img_l[:, :, 2].std()) # type: ignore

def merge_inference(
    outdir: Path, testdir_regex: str = "test_fold*", savedir_name: str = "test"
):
    """
    Merge inference results from multiple directories into a single directory.

    Args:
        outdir (Path): The output directory where the merged inference results will be saved.
        testdir_regex (str, optional): The regular expression pattern to match the test directories to merge. Defaults to "test_fold*".
        savedir_name (str, optional): The name of the directory where the merged results will be saved inside the `outdir`. Defaults to "test".

    Returns:
        None
    """
    savedir = outdir / savedir_name
    testdir_list = sorted(outdir.glob(testdir_regex))
    print(f"target directories: {testdir_list}")
    # 本当はすべてのdirのファイル名が同じであることを確認したい
    # 今回は省略
    target_files = natsorted(testdir_list[0].glob("*.tif")) # type: ignore

    if savedir.exists():
        shutil.rmtree(savedir, ignore_errors=True)
    savedir.mkdir(parents=True, exist_ok=True)
    for target_file in tqdm(target_files, leave=False, desc="merging inference"):
        preds = []
        for testdir in testdir_list:
            preds.append(tifffile.imread(testdir / target_file.name))
        preds = np.array(preds)
        preds = np.mean(preds, axis=0)
        preds = np.round(preds, decimals=0).astype(np.uint8)
        tifffile.imwrite(
            savedir / target_file.name,
            preds,
        )


def slice_img(img: np.ndarray, slicing_size: int = 130, sliding_stride: int = 130) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Slice an image into smaller patches.

    Args:
        img (np.ndarray): The image to slice.
        slicing_size (int): The size of the patches to slice the image into.
        sliding_stride (int): The stride of the sliding window.

    Returns:
        Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]: A tuple containing:
            - A list of np.ndarray: the sliced image patches.
            - A list of Tuple[int, int, int, int]: their corresponding indices (e.g., x_start, y_start, x_end, y_end).
    """
    # Assuming img_patches.extract_patches returns a tuple: (list_of_patches, list_of_indices)
    # If the types returned by extract_patches are more complex, adjust the List contents accordingly.
    sliced_imgs, indices = img_patches.extract_patches(data=img, patchsize=slicing_size, overlap=0.2, stride=sliding_stride)
    return sliced_imgs, indices


def merge_img(img_list: list[np.ndarray], indices: list[Tuple[int, int, int, int]]) -> np.ndarray:
    """
    Merge a list of images into a single image.

    Args:
        img_list (List[np.ndarray]): The list of images to merge.
        indices (list[Tuple[int, int, int, int]]): The indices of the images in the list.

    Returns:
        np.ndarray: The merged image.
    """
    return img_patches.merge_patches(img_list, indices)

def resize_img(img: np.ndarray, size: Tuple[int, int] = (130, 130)) -> np.ndarray:
    """
    Resize an image to a specified size using OpenCV.

    Args:
        img (np.ndarray): The input image to be resized.
        size (Tuple[int, int]): The target size for the image. Defaults to (130, 130).

    Returns:
        np.ndarray: The resized image.

    Notes:
        This function uses OpenCV's INTER_AREA interpolation for resizing.
    """
    if img.shape[0] > size[0] or img.shape[1] > size[1]:
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return img