import os
import shutil
import requests
from tqdm import tqdm
from typing import Tuple, List
import zipfile
import random
import tifffile
from bs4 import BeautifulSoup

import cv2
import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.seed import seed_everything
from transformers import AutoImageProcessor


#######################################
# Dataset
#######################################
class CustomDataset(Dataset):
    def __init__(self,
        cfg,
        tf_dict: dict,
        df: pd.DataFrame,
        phase: str = "train",
        epoch: int = 0,
        is_path: bool = False,
    ):
        """pytorch dataset for Solafune Super Resolution 2023 data."""
        self.df = df
        self.cfg = cfg
        self.phase = phase
        self.epoch = epoch
        self.is_path = is_path
        self.tf_dict = tf_dict
        self.transform = self.tf_dict[self.phase]
        self.processor = AutoImageProcessor.from_pretrained(cfg.model_name)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(
        self, index: int
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, str]:
        """
        Returns:
            in_pad: Tensor ... 低解像度画像 (channel, W_l, H_l)
            mask: Tensor ... 高解像度画像 (channel, W_h, H_h).
            row.path_low: str ... low resolution image path
        """
        row = self.df.iloc[index]

        img_l = tifffile.imread(row.path_low)

        if self.phase == "test":
            img = self.transform(image=img_l)["image"]
            img = self.processor(img, return_tensors="pt")["pixel_values"].squeeze(0)
            return img, row.path_low

        elif self.phase == "train" or self.phase == "val":
            img_h = tifffile.imread(row.path_high)

            # augmentation
            if self.phase == "train" and self.cfg.augmentation_end_epoch <= self.epoch:
                self.transform = self.tf_dict["train_augless"]

            transformed = self.transform(image=img_l, mask=img_h)

            # cutmixは20epochまで
            if self.phase == "train" and self.epoch < self.cfg.cutmix_end_epoch:
                ref_row = self.df.sample(1).iloc[0]
                ref_img_l = tifffile.imread(ref_row.path_low)
                ref_img_h = tifffile.imread(ref_row.path_high)
                ref_transformed = self.transform(image=ref_img_l, mask=ref_img_h)

                transformed["image"], transformed["mask"] = cutmix(
                    transformed["image"],
                    transformed["mask"].permute(2, 0, 1),
                    ref_transformed["image"],
                    ref_transformed["mask"].permute(2, 0, 1),
                )
                transformed["mask"] = transformed["mask"].permute(1, 2, 0)

            img = transformed["image"]
            mask = transformed["mask"].permute(2, 0, 1)

            img = self.processor(img, return_tensors="pt")["pixel_values"].squeeze(0)

            if self.is_path:
                return img, mask, row.path_low
            return img, mask  # in_pad ... 低解像度画像 mask... 高解像度画像
        else:
            raise NotImplementedError("phase must be train, val or test.")

class CustomDataLoaderModule(LightningDataModule):
    def __init__(
        self,
        cfg,
        tf_dict: dict,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
    ):
        """pytorch lightning datamodeule for Solafune Super Resolution 2023 data."""
        super().__init__()
        self.df_train = df_train
        self.df_val = df_val
        self._cfg = cfg
        self.tf_dict = tf_dict

    def train_dataloader(self) -> DataLoader:
        """
        Returns the training dataloader for the model.

        Parameters:
            None.

        Returns:
            DataLoader: The training dataloader.

        Raises:
            None.
        """
        epoch = self.trainer.current_epoch # type: ignore
        if self._cfg.extra_dataset:
            if epoch <= self._cfg.extra_dataset_end_epoch:
                data = self.df_train
            else:
                data = self.df_train.query("fold != -1")
        else:
            data = self.df_train
        dataset = CustomDataset(df=data, phase="train", epoch=epoch, is_path=True, cfg=self._cfg, tf_dict=self.tf_dict)
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self) -> DataLoader:
        """
        Generates the validation data loader for the current epoch.

        Returns:
            DataLoader: The validation data loader.
        """
        epoch = self.trainer.current_epoch # type: ignore
        dataset = CustomDataset(df=self.df_val, phase="val", epoch=epoch, is_path=True, cfg=self._cfg, tf_dict=self.tf_dict)
        
        return DataLoader(dataset, **self._cfg.val_loader)

def cutmix(LR: torch.Tensor, HR: torch.Tensor, 
           refLR: torch.Tensor, refHR: torch.Tensor,
           p: float = 1.0, alpha: float = 0.7) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a cutmix of the given low-resolution (LR) and high-resolution (HR) tensors.

    Args:
        LR (torch.Tensor): The low-resolution input tensor.
        HR (torch.Tensor): The high-resolution input tensor.
        refLR (torch.Tensor): The reference low-resolution tensor.
        refHR (torch.Tensor): The reference high-resolution tensor.
        p (float, optional): The probability of applying cutmix. Defaults to 1.0.
        alpha (float, optional): The alpha value for generating the random normal distribution. Defaults to 0.7.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the cutmixed low-resolution and high-resolution tensors.
    """

    if torch.rand(1) >= p:
        return LR, HR

    scale = HR.size(1) // LR.size(1)
    v = np.random.normal(alpha, 0.01)
    h, w = LR.size()[1:]
    ch, cw = int(h * v), int(w * v)  # type: ignore

    fcy, fcx = np.random.randint(0, h - ch + 1), np.random.randint(0, w - cw + 1)
    tcy, tcx = np.random.randint(0, h - ch + 1), np.random.randint(0, w - cw + 1)

    LR[:, tcy : tcy + ch, tcx : tcx + cw] = refLR[:, fcy : fcy + ch, fcx : fcx + cw]
    HR[:, tcy * scale : (tcy + ch) * scale, tcx * scale : (tcx + cw) * scale] = refHR[:, fcy * scale : (fcy + ch) * scale, fcx * scale : (fcx + cw) * scale]

    return LR, HR

def set_seed(seed) -> None:
    """
    Set the seed for random number generation.

    Parameters:
        seed (int): The seed value to be set.

    Returns:
        None
    """
    # 乱数のシードを設定
    seed_everything(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)