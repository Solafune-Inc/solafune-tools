from __future__ import annotations
from pathlib import Path
import gc
import shutil
from colorama import init, Fore

# Initializes Colorama
init(autoreset=True)

from tqdm import tqdm

tqdm.pandas()
import numpy as np
import pandas as pd

pd.set_option("display.max_rows", 30)
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 1000)
# pd.options.display.max_colwidth = 250
# pd.options.display.max_rows = 30
import tifffile

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from transformers import (
    Swin2SRForImageSuperResolution,
)
from pytorch_lightning import LightningModule
import lion_pytorch as lion
import torch.optim as optim

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from solafune_tools.osm.super_resolution.config import CFG
    from solafune_tools.osm.super_resolution.common_utils import AverageMeter, RemainedTimeEstimator, get_total_n_steps
except:
    from config import CFG
    from common_utils import AverageMeter, RemainedTimeEstimator, get_total_n_steps
# ====================================================
# AWP
# ====================================================
class AWP:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        adv_param="weight",
        adv_lr=1.0,
        adv_eps=0.01,
    ):
        """
        Initializes the class with the given parameters.

        Args:
            model (object): The model to be used for training.
            criterion (object): The criterion to be used for calculating the loss.
            optimizer (object): The optimizer to be used for updating the model parameters.
            adv_param (str, optional): The name of the parameter to be adversarially trained. Defaults to "weight".
            adv_lr (float, optional): The learning rate for the adversarial training. Defaults to 1.0.
            adv_eps (float, optional): The epsilon value for the adversarial training. Defaults to 0.01.

        Returns:
            None

        Note:
            This function initializes the class with the specified parameters and sets the attributes accordingly.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, inputs, label):
        """
        Attack the model by modifying it in a bad direction, and calculate the adversarial loss.

        Parameters:
            inputs: The input data to be fed to the model.
            label: The target label for the input data.

        Returns:
            adv_loss: The adversarial loss calculated using the model's predictions and the target label.

        Notes:
            - This function modifies the model by calling the _save() and _attack_step() methods.
            - The model's predictions are obtained by calling the model(inputs) method.
            - The adversarial loss is calculated using the criterion(y_preds, label) method.
            - The optimizer's gradients are set to zero using the optimizer.zero_grad() method.
        """
        self._save()
        self._attack_step()  # モデルを近傍の悪い方へ改変
        y_preds = self.model(inputs)  # SRModelの前提
        adv_loss = self.criterion(y_preds, label)
        self.optimizer.zero_grad()
        return adv_loss

    def _attack_step(self):
        """
        Applies an attack step to the model's parameters.

        This function iterates over all named parameters of the model and applies
        an attack step to the parameters that meet the following conditions:
        - The parameter requires gradients.
        - The parameter has a gradient value.
        - The parameter name contains the specified `adv_param` string.

        For each eligible parameter, the function calculates the L2 norm of the gradient
        and the L2 norm of the parameter data. If the gradient norm is non-zero and not NaN,
        the function calculates the attack step `r_at` using the specified `adv_lr` learning rate,
        the gradient, and the parameter data. The attack step is then added to the parameter data,
        and the resulting value is clipped between the specified `backup_eps` minimum and maximum values.

        Parameters:
            None

        Returns:
            None

        Notes:
            - This function assumes that the model's parameters have already been initialized.
            - The attack step is calculated as `r_at = adv_lr * param.grad / (norm1 + e) * (norm2 + e)`,
              where `e` is a small constant to avoid division by zero.
            - The resulting parameter value is clipped between the `backup_eps` minimum and maximum values.
        """
        e = 1e-6
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    # 直前に損失関数に通してパラメータの勾配を取得できるようにしておく必要あり
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]),
                        self.backup_eps[name][1],
                    )
                # param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        """
        Saves the model parameters that require gradients and have a name containing the specified `adv_param`.

        This function iterates over the named parameters of the model. For each parameter, it checks if it requires gradients and if it has a name containing the `adv_param`. If both conditions are met, it saves the parameter's current value as a backup. It also calculates the `grad_eps` value as the product of `adv_eps` and the absolute value of the parameter's current value. The function then saves the lower and upper bounds of the parameter's value by subtracting and adding `grad_eps` respectively from the backup value.

        Parameters:
        - None

        Returns:
        - None

        Notes:
        - This function is used for saving model parameters that are relevant to adversarial training or perturbation generation.
        """
        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and param.grad is not None
                and self.adv_param in name
            ):
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self):
        """
        Restores the model parameters to their original values.

        This function iterates over the named parameters of the model and checks if they are present in the backup dictionary. If a parameter is found in the backup, its data is restored to the original value. The backup dictionary is then cleared.

        Parameters:
        - None

        Returns:
        - None

        Notes:
        - This function assumes that the model has been previously backed up using the `_backup()` method.
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

#######################################
# SSIM Loss
#######################################
class SSIMLoss(nn.Module):
    def __init__(self, kernel_size: int = 7, sigma: float = 1.5):
        """
        Initializes a new instance of the class.
        
        Args:
            kernel_size (int): The size of the kernel used for the SSIM filter. Default is 7.
            sigma (float): The standard deviation of the kernel used for the SSIM filter. Default is 1.5.
        
        Notes:
            - The `kernel_size` parameter determines the size of the kernel used for the SSIM filter.
            - The `sigma` parameter determines the standard deviation of the kernel used for the SSIM filter.

        Returns:
            None
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.ssim = SSIM(kernel_size=self.kernel_size, sigma=self.sigma)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return 1 - self.ssim(x, y)

#######################################
# define Model
#######################################


class SRModel(LightningModule):
    def __init__(self, cfg, fold: int):
        """
        Initializes a new instance of the class.
        
        Args:
            cfg (DictConfig): The configuration object.
            fold (int): The fold number.
        
        Notes:
            - The `cfg` parameter is the configuration object.
            - The `fold` parameter is the fold number.
        
        Returns:
            None
        """
        super().__init__()
        self.automatic_optimization = False
        self.cfg = cfg
        self.__build_model()
        self.score_max = 0.0
        self.fold = fold  # 何fold目のモデルか？
        self.remained_time_estimator = RemainedTimeEstimator()

        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        # self.criterion = SSIMpixLoss()
        self.criterion = SSIMLoss()
        self.losses = AverageMeter()

    def __build_model(self):
        """
        This function builds the model.

        Parameters:
        - None
        
        Returns:
        - None
        
        Notes:
        - This function builds the model based on the configuration object.
        """
        self.backbone = Swin2SRForImageSuperResolution.from_pretrained(self.cfg.model_name)

    def forward(self, x: Tensor):
        """
        This function is the forward pass of the model.

        Args:
            x (Tensor): The input tensor.
        
        Returns:
            Tensor: The output tensor.
        
        Notes:
        - This function is the forward pass of the model.
        """
        f = self.backbone(x)["reconstruction"]  # type: ignore
        f = f[:, :, 0:self.CROP_SIZE, 0:self.CROP_SIZE]
        # assert f.shape == (x.shape[0], 3, self.CROP_SIZE, self.CROP_SIZE)
        f = TF.resize(f, size=650, interpolation=T.InterpolationMode.BICUBIC) # type: ignore
        f = torch.clamp(f, min=0.0, max=1.0)
        f = f * 255
        return f

    def on_train_epoch_start(self):
        """
        Initializes the training epoch by printing a message if the current epoch is greater than or equal to the start epoch specified in the configuration. Then, it creates an instance of the AWP (Adversarial Weight Perturbation) class, passing in the necessary parameters.

        Parameters:
            None

        Returns:
            None

        Notes:
            - The AWP class is used for adversarial training.
            - The AWP instance is stored in the `self.awp` attribute.
        """
        if self.cfg.awp.start_epoch <= self.current_epoch: # type: ignore
            print(f"AWP training with epoch {self.current_epoch}")

        self.awp = AWP(
            self,
            self.criterion,
            self.optimizers(use_pl_optimizer=True),
            adv_lr=self.cfg.awp.lr, # type: ignore
            adv_eps=self.cfg.awp.eps, # type: ignore
        )

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx
        ) -> dict[str, Tensor]:
        """
        Runs a training step on a batch of data.

        Args:
            batch (tuple[Tensor, Tensor]): A tuple containing the input and target tensors.
            batch_idx: The index of the batch.

        Returns:
            dict[str, Tensor]: A dictionary containing the loss value.

        Notes:
            - The input tensors should be of type float.
            - The loss value is calculated using the criterion function.
            - Gradients are zeroed using the optimizer's zero_grad() method.
            - Backward pass is performed using the manual_backward() method.
            - Gradients are clipped using the clip_gradients() method.
            - If the AWP start epoch is less than or equal to the current epoch, AWP attack is applied.
            - Optimizer step is performed using the optimizer's step() method.
            - Learning rate scheduler step is performed using the lr_schedulers() method.
            - The loss value is updated in the losses object.
            - Training progress information is printed if the current batch index plus one is a multiple of the print log interval.
        """
        opt = self.optimizers(use_pl_optimizer=True)

        ret = {}
        imgs_l, imgs_h, path_list = batch # type: ignore
        imgs_l, imgs_h = imgs_l.float(), imgs_h.float()
        batch_size = imgs_l.shape[0]
        preds_h = self.forward(imgs_l)
        loss = self.criterion(preds_h, imgs_h)

        opt.zero_grad()  # type: ignore
        self.manual_backward(loss)
        self.clip_gradients(
            opt, gradient_clip_val=self.cfg.grad_clip, gradient_clip_algorithm="norm"  # type: ignore
        )

        if self.cfg.awp.start_epoch <= self.current_epoch:  # AWPを適応 # type: ignore
            adv_loss = self.awp.attack_backward(imgs_l, imgs_h)
            self.manual_backward(adv_loss)
            self.awp._restore()  # WPする前のモデルに復元

        opt.step()  # type: ignore
        sch = self.lr_schedulers()
        sch.step()  # type: ignore

        ret.update({"loss": loss})
        self.losses.update(loss.item(), batch_size)  # lossの平均を計算するためのもの
        if (batch_idx + 1) % self.cfg.print_log_interval == 0:
            # get epoch n steps
            epoch_n_steps = len(self.trainer.train_dataloader) # type: ignore
            print(
                f"Epoch:{self.current_epoch}",
                f"Step:{batch_idx + 1}/{epoch_n_steps}",
                f"Loss:{self.losses.val:.4f}, Loss (avg on epoch):{self.losses.avg:.4f}",
                f"LR:{self.trainer.optimizers[0].param_groups[0]['lr']:.6f}",
                f"ELAPSED:{self.remained_time_estimator(batch_idx / epoch_n_steps)}",
            )
        # del loss, preds_h, imgs_l, imgs_h
        # gc.collect()
        # torch.cuda.empty_cache()
        return ret

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx
    ) -> dict[str, Tensor]:
        """
        Performs a validation step on the given batch.

        Args:
            batch (tuple[Tensor, Tensor]): A tuple containing the input images (imgs_l),
                target images (imgs_h), and path list (path_list).
            batch_idx: The index of the current batch.

        Returns:
            dict[str, Tensor]: A dictionary containing the predicted high-resolution images
                (preds_h), the path list (path_list), the loss value (loss), the mean absolute
                error (mae), the mean squared error (mse), the score, and the batch size.

        Notes:
            - The input images (imgs_l) and target images (imgs_h) are expected to be float tensors.
            - The predicted high-resolution images (preds_h) are rounded to the nearest integer value.
            - The loss is calculated using the criterion function.
            - The score is calculated as 1 minus the loss value.
            - The mean absolute error (mae) is calculated using the mae function.
            - The mean squared error (mse) is calculated using the mse function.
            - The batch size is determined by the number of input images (imgs_l) in the batch.
            - The loss value is updated in the losses object.
            - The validation step information is printed if the current batch index plus one is a
              multiple of the print log interval.
        """
        ret = {} 
        imgs_l, imgs_h, path_list = batch # type: ignore
        imgs_l, imgs_h = imgs_l.float(), imgs_h.float()
        batch_size = imgs_l.shape[0]

        preds_h = self.forward(imgs_l)
        preds_h = torch.round(preds_h, decimals=0)
        ret["preds_h"] = preds_h
        ret["path_list"] = list(path_list)

        loss = self.criterion(preds_h, imgs_h)

        # mertics
        score = 1 - self.criterion(preds_h, imgs_h).detach().cpu()
        mse = self.mse(preds_h, imgs_h).detach().cpu()
        mae = self.mae(preds_h, imgs_h).detach().cpu()

        ret.update({
            "loss": loss, 
            "mae": mae, 
            "mse": mse, 
            "score": score,
            "batch_size": batch_size
            })
        # self.log_dict(ret, on_step=True, on_epoch=True, logger=True)
        self.losses.update(loss.item(), batch_size)

        if (batch_idx + 1) % self.cfg.print_log_interval == 0:
            epoch_n_steps = (
                self.trainer.estimated_stepping_batches
                // 2
                // self.cfg.epoch
                // (self.cfg.folds - 1)
                + 1
            )
            print(
                f"Epoch:{self.current_epoch}",
                f"Step:{batch_idx + 1}/{epoch_n_steps}",
                f"Loss:{self.losses.val:.4f}, Loss (avg on epoch):{self.losses.avg:.4f}",
                f"LR:{self.trainer.optimizers[0].param_groups[0]['lr']:.6f}",
                f"ELAPSED:{self.remained_time_estimator(batch_idx / epoch_n_steps)}",
            )
        return ret

    def training_epoch_end(self, outputs: list[dict[str, Tensor]]):
        """
        Performs a training epoch end. This method is called at the end of each training epoch.
        """
        # print("this is trainig_epoch_end")
        mode = "train"
        self.__share_epoch_end(outputs, mode)
        losses = []
        for out in outputs:
            loss = out["loss"].detach().cpu()
            losses.append(loss)
        losses = np.mean(losses)
        self.log(f"{mode}/loss", losses) # type: ignore

        gc.collect()

    def validation_epoch_end(self, outputs: list[dict[str, Tensor]]):
        """
        Performs a validation epoch end. This method is called at the end of each validation epoch.
        """

        # print("this is validation_epoch_end")
        mode = "val"
        self.__share_epoch_end(outputs, mode)
        scores, maes, mses, losses, batch_sizes = [], [], [], [], []
        for out in outputs:
            score, mae, mse, loss, batch_size = (
                out["score"],
                out["mse"],
                out["mae"],
                out["loss"].detach().cpu(),
                out["batch_size"]
            )
            scores.append(score)
            losses.append(loss)
            maes.append(mae)
            mses.append(mse)
            batch_sizes.append(batch_size)

        if self.cfg.batch_size_log:
            scores = np.average(scores, weights=batch_sizes)
            losses = np.average(losses, weights=batch_sizes)
            mses = np.average(mses, weights=batch_sizes)
            maes = np.average(maes, weights=batch_sizes)
        else:
            scores = np.mean(scores)
            losses = np.mean(losses)
            mses = np.mean(mses)
            maes = np.mean(maes)

        self.log(f"{mode}/score", scores) # type: ignore
        self.log(f"{mode}/loss", losses) # type: ignore
        self.log(f"{mode}/mse", mses)  # type: ignore
        self.log(f"{mode}/mae", maes)  # type: ignore

        if self.score_max < scores: # type: ignore
            self.score_max = scores
            # 以下oofの保存
            print(
                Fore.CYAN
                + f"[Fold{self.fold} epoch{self.current_epoch}] Save Best Out-Of-Folds with Score: {self.score_max:.4f}"
            )
            for i, out in enumerate(outputs):
                preds_h = (
                    out["preds_h"]
                    .detach()
                    .cpu()
                    .numpy()
                    .transpose(0, 2, 3, 1)
                    .astype(np.uint8)
                )
                path_low_list = out["path_list"]
                for a_image, path_low in zip(preds_h, path_low_list):
                    save_dir = self.cfg.outdir / f"oof_data_fold{self.fold}"
                    if not save_dir.exists():
                        save_dir.mkdir(parents=True)
                    filename = Path(path_low).name.replace("low", "oof") # type: ignore
                    tifffile.imwrite(
                        save_dir / filename,
                        a_image,
                    )

        gc.collect()

    def __share_epoch_end(self, outputs: list[dict[str, Tensor]], mode: str):
        """
        Share the epoch end information.

        Args:
            outputs (list[dict[str, Tensor]]): The output of the model.
            mode (str): The mode of the epoch.

        Returns:
            None
        """
        
        self.losses = AverageMeter()  # lossの平均初期化
        self.remained_time_estimator = RemainedTimeEstimator()  # 時間計測初期化

    def configure_optimizers(self) -> tuple[list["Optimizer"], list["_LRScheduler"]]: # type: ignore
        """
        Configure the optimizers.

        Returns:
            tuple[list["Optimizer"], list["_LRScheduler"]]: The optimizers and schedulers.
        """
        # get total steps
        total_steps = get_total_n_steps(self.trainer.datamodule, self.cfg.epoch) # type: ignore

        optimizer = eval(self.cfg.optimizer.name)(
            self.parameters(), **self.cfg.optimizer.params
        )
        if self.cfg.use_total_steps:
            scheduler = {
            "scheduler": eval(self.cfg.scheduler.name)(
                optimizer,
                T_0=int(total_steps * 1.03),
                **self.cfg.scheduler.params,
                ),
            "interval": "step",
            }
        else:
            scheduler = { 
                "scheduler": eval(self.cfg.scheduler.name)(
                    optimizer,
                    T_0=int(self.trainer.estimated_stepping_batches * 1.0),
                    **self.cfg.scheduler.params,
                ),
                "interval": "step",
            }
            
        return [optimizer], [scheduler]
    
#######################################
# inference for test data
#######################################
@torch.inference_mode()
def inferece_fn(model: nn.Module, test_loader: DataLoader, save_dir: Path, cfg) -> None:
    """
    Runs inference on a given model using a test data loader and saves the results to a specified directory.
    
    Args:
        model (nn.Module): The model to be used for inference.
        test_loader (DataLoader): The data loader containing the test data.
        save_dir (Path): The directory where the inference results will be saved.
        cfg: The configuration object containing additional settings.
        
    Returns:
        None
    """
    
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True)
    model.to(cfg.device)
    for (image, paths) in tqdm(test_loader, leave=False, desc="inferrencing for test"):
        image = image.to(cfg.device)
        preds = model(image)
        preds = torch.round(preds, decimals=0)
        preds = preds.detach().cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)

        # per batch
        for a_image, path in zip(preds, paths):
            filename = Path(path).name.replace("_low.tif", "_answer.tif")
            tifffile.imwrite(
                save_dir / filename,
                a_image,
            )

#######################################
# cv score
#######################################
def valid_fn(cfg,oof_file_paths: list[Path]) -> float:
    """
    Calculates the validation score for the given out-of-fold (oof) file paths.

    Parameters:
        cfg (Config): The configuration object containing the device information.
        oof_file_paths (list[Path]): A list of file paths pointing to the oof files.

    Returns:
        float: The calculated validation score.
    """
    ssim = SSIMLoss()
    ssim.to(cfg.device)
    scores = []
    for oof_file in tqdm(oof_file_paths, leave=True, desc="calc cv score..."):
        oof = tifffile.imread(oof_file)
        truth = tifffile.imread(
            cfg.ROOT_TRAIN / oof_file.name.replace("_oof.tif", "_high.tif")
        )
        truth = (
            torch.tensor(truth, dtype=torch.float32)
            .permute(2, 0, 1)
            .to(cfg.device)
            .unsqueeze(dim=0)
        )
        oof = (
            torch.tensor(oof, dtype=torch.float32)
            .permute(2, 0, 1)
            .to(cfg.device)
            .unsqueeze(dim=0)
        )
        with torch.no_grad():
            score = 1 - ssim(truth, oof)
        scores.append(score.detach().cpu().numpy())
    score = np.mean(scores)
    return score  # type: ignore