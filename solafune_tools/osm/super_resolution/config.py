import os
from pathlib import Path
import torch
from typing import Any, Union, List

#######################################
# config
#######################################

OUTPUT_DIR = Path("working/")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

class CFG(object):
    def __init__(self, 
                 debug: bool, datasets_dir: str,
                 batch_size: int, grad_clip: float, 
                 learning_rate: float, augmentation_end_epoch: int,
                 cutmix_end_epoch: int, extra_dataset: bool,
                 extra_dataset_end_epoch: int, train_loader_workers: int, 
                 val_loader_workers: int, awp_start_epoch: int,
                 batch_size_log: bool, use_total_steps:bool,
                 gpus: Union[int, str, List] = 1, strategy: Union[str, bool] = False):
        # BASIC
        self.debug: bool = debug  # True
        self.debug_sample: int = 64
        self.wandb: bool = False  # True if not debug else False
        self.wandb_alert_freq: str = "every" #  "every" or "last" or None
        self.save_model_to_gcs: bool = False  # True
        self.shutdown: bool = True if not self.debug else False
        self.folds: int = 4
        self.trn_folds: list[int] = [0, 1, 2, 3]
        self.seed: int = 417
        self.eps: float = 1e-12
        self.outdir: Path = OUTPUT_DIR
        self.ckpt_stem: str = "trained_model"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # DATA
        self.DATA_ROOT: Path = Path(datasets_dir)
        self.ROOT_TRAIN: Path = self.DATA_ROOT / "train"
        self.ROOT_TEST: Path = self.DATA_ROOT / "test"
        self.ROOT_SAMPLE: Path = self.DATA_ROOT / "uploadsample"

        self.preprocess: dict[str, int] = {"input_size": 130, "upscale": 5, "output_size": 650}

        # TRAIN
        self.epoch: int = 40 if not self.debug else 2  # 100
        self.batch_size: int = batch_size
        self.grad_clip: float = grad_clip

        if strategy == False:
            if isinstance(gpus, list):
                if len(gpus) > 1:
                    gpus = 1
            elif gpus > 1: # type: ignore
                gpus = 1

        if gpus:
            accelerator = "gpu"
            devices = gpus if isinstance(gpus, int) else len(gpus)

        else:
            accelerator = "cpu"
            devices = 1

        if strategy:
            self.trainer: dict[str, Any] = {
                "accelerator": accelerator,
                "devices" : devices,
                "accumulate_grad_batches": 1,  # 勾配累積 #経験上増やして良くなった試しはない(基本的に悪化する。生バッチ増やしたほうがいい)
                "fast_dev_run": False,  # Trueでデバッグ用モード。1epochで終了
                "num_sanity_val_steps": 0,  # デバッグ関係の引数
                # "log_every_n_steps": 5,  # 何stepごとにログを吐き出すか # pytorch lightning における self.logでの操作が難しい
                "check_val_every_n_epoch": 1,  # 何epochごとにvalを行うか
                "val_check_interval": 1.0,
                # "precision": 16,  # 16bitで計算
                # "gradient_clip_val": 25.0, # automatic_optimization=Trueのときに有効
                # "gradient_clip_algorithm": "value", # automatic_optimization=Trueのときに有効
                "enable_progress_bar": False,  # wandbのログが死ぬので切る
                "reload_dataloaders_every_n_epochs": 1,
                "strategy": strategy
            }
        else:
            self.trainer: dict[str, Any] = {
                "accelerator": accelerator,
                "devices" : devices,
                "accumulate_grad_batches": 1,  # 勾配累積 #経験上増やして良くなった試しはない(基本的に悪化する。生バッチ増やしたほうがいい)
                "fast_dev_run": False,  # Trueでデバッグ用モード。1epochで終了
                "num_sanity_val_steps": 0,  # デバッグ関係の引数
                # "log_every_n_steps": 5,  # 何stepごとにログを吐き出すか # pytorch lightning における self.logでの操作が難しい
                "check_val_every_n_epoch": 1,  # 何epochごとにvalを行うか
                "val_check_interval": 1.0,
                # "precision": 16,  # 16bitで計算
                # "accelerator": "gpu",
                # "devices": len(gpus),
                # "gradient_clip_val": 25.0, # automatic_optimization=Trueのときに有効
                # "gradient_clip_algorithm": "value", # automatic_optimization=Trueのときに有効
                "enable_progress_bar": False,  # wandbのログが死ぬので切る
                "reload_dataloaders_every_n_epochs": 1,
            } 

        self.print_log_interval: int = 50 if not self.debug else 5  # 何stepごとにログを吐き出すか

        # Based on our experience, a suitable learning rate for Lion is typically 10x smaller than that for AdamW, although sometimes a learning rate that is 3x smaller may perform slightly better.
        self.optimizer: dict[str, Any] = {
            # "name": "optim.AdamW",
            # "name": "bnb.optim.AdamW32bit",
            "name": "lion.Lion",
            "params": {
                "lr": learning_rate,
            },
        }
        self.scheduler: dict[str, Any] = {
            "name": "optim.lr_scheduler.CosineAnnealingWarmRestarts",
            "params": {
                "eta_min": 1e-7,
            },
        }
        self.model_name = "caidas/swin2SR-classical-sr-x4-64"
        self.train_loader: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "pin_memory": True,  # 今画像なのでpin_memoryするとメモリが死にそう？
            "drop_last": True,
            "num_workers": train_loader_workers,  # もしデッドロックしたらここを1に
        }
        self.augmentation_end_epoch: int = augmentation_end_epoch
        self.cutmix_end_epoch: int = cutmix_end_epoch
        self.extra_dataset: bool = extra_dataset 
        self.extra_dataset_end_epoch: int = extra_dataset_end_epoch
        self.val_loader: dict[str, Any] = {
            "batch_size": 2 * self.batch_size,
            "shuffle": False,
            "pin_memory": True,
            "drop_last": False,
            "num_workers": val_loader_workers,  # もしデッドロックしたらここを1に
        }
        self.awp: dict[str, Any] = {"lr": 0.1, "eps": 0.001, "start_epoch": awp_start_epoch}  # 0スタート

        # logging
        self.project: str = "solafune-SR-2023"
        self.runname: str = "training_notebook.ipynb" #Path(__file__).stem
        self.group: str = self.model_name

        # post info
        self.augmentation: str = ""
        self.fold: int = -1
        self.batch_size_log: bool = batch_size_log
        
        # Optimizer config
        self.use_total_steps: bool = use_total_steps

        if self.debug:
            self.epoch = 2
            self.group = "DEBUG"
            self.trn_folds = [0, 2]
            self.augmentation_end_epoch = 1  # 0スタートでこのときには抜く
            self.cutmix_end_epoch = 1  # 0スタートでこのときには抜く
            self.awp["start_epoch"] = 1  # 0スタートでこのときから入れる
            self.extra_dataset_end_epoch = 1
        