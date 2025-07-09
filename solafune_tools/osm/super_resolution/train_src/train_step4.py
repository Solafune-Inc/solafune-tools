#######################################
# import
#######################################
from __future__ import annotations
from collections import defaultdict
import warnings
from glob import glob
import gc
from natsort import natsorted
from colorama import init, Fore
import os

from typing import List, Union

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
#from IPython.display import display
from box import Box

from sklearn.model_selection import KFold
import torch

from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from pytorch_lightning import Trainer
from pytorch_lightning import callbacks
from pytorch_lightning.loggers import TensorBoardLogger

from config import CFG
from preprocess import CustomDataset, CustomDataLoaderModule, set_seed
from common_utils import OSMResourceDownloader, load_image_info, delete_if_file_exists, merge_inference
from model import SRModel, inferece_fn, valid_fn

warnings.filterwarnings("ignore")

# 高速化option
torch.autograd.set_detect_anomaly(False) # type: ignore
torch.backends.cudnn.benchmark = True

import os, sys, argparse, traceback

def is_dataset_folder_exist():
    train_dir_exist, test_dir_exist  = os.path.isdir("datasets/train"), os.path.isdir("datasets/test")
    if train_dir_exist and test_dir_exist:
        return True
    else: return False

def train(using_own_dataset : bool = False,  gpus: Union[int, str, List] = 1, debug: bool = False,  strategy: Union[str, bool] = False):
    """
    Trains a model using the given dataset.

    Parameters:
        None

    Returns:
        None
    """
    dataset = OSMResourceDownloader()
    if using_own_dataset:
        dataset_folder_check = is_dataset_folder_exist()
        if not dataset_folder_check:
            sys.exit("""
                     Exiting the training Script!
                     It seems you are not following the project tree structure for the dataset or perhaps you have not prepare the dataset structure correctly
                     Please prepare your dataset stucture correctly like this bellow:
                        datasets
                            ├── test
                            │   ├── *_low.tif 
                            │   └── ... 
                            └── train
                                ├── *_test.tif
                                ├── *_train.tif
                                └── ...
                     """)
        print("Using own dataset as main dataset for training instead competition dataset.")
    else:
        dataset.base_dataset_download(redownload=False)
    
    #dataset.download_extra_datasets()

    # Box

    cfg = Box({k: v for k, v in dict(vars(CFG(debug=debug, datasets_dir=dataset.datasets_dir,
                                              batch_size=1, grad_clip=100.0,
                                              learning_rate=3e-5, augmentation_end_epoch=27,
                                              cutmix_end_epoch=22, extra_dataset=True,
                                              extra_dataset_end_epoch=17, train_loader_workers=2,
                                              val_loader_workers=2, awp_start_epoch=35,
                                              batch_size_log=True, use_total_steps=True,
                                              gpus=gpus, strategy=strategy))).items() if "__" not in k})
    
    # Set Random Seed
    set_seed(cfg.seed)

    #######################################
    # Preprocess
    # それなりに時間がかかる処理なので処理済みのものを持ってくる形式にしたほうが良い気がする
    #######################################

    PATHS_HIGH_TRAIN = sorted(glob(f"{cfg.ROOT_TRAIN}/train_*_high.tif"))
    PATHS_LOW_TRAIN = sorted(glob(f"{cfg.ROOT_TRAIN}/train_*_low.tif"))

    df = pd.DataFrame({"path_high": PATHS_HIGH_TRAIN, "path_low": PATHS_LOW_TRAIN}) # type: ignore
    #display(df.sample(4))

    meta_columns = [
        f"{reso}_{feat}"
        for reso in ["high", "low"]
        for feat in [
            "hight",
            "width",
            "dtype",
            "r_mean",
            "r_std",
            "g_mean",
            "g_std",
            "b_mean",
            "b_std",
        ]
    ]

    df[meta_columns] = df.progress_apply(load_image_info, axis=1, result_type="expand")  # type: ignore
    # fold
    n_fold = np.zeros(len(df))
    folder = KFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    for fold, (_, val_idx) in enumerate(folder.split(range(len(df)))):  # type: ignore
        n_fold[val_idx] = fold
    df["fold"] = n_fold.astype(np.uint16)
    #display(df.head(6))
    df.to_csv(cfg.DATA_ROOT / "train.csv", index=False)

    PATHS_HIGH_TRAIN = sorted(glob(os.path.join(cfg.DATA_ROOT, "train/train_*_high.tif")))
    PATHS_LOW_TRAIN = sorted(glob(os.path.join(cfg.DATA_ROOT,"train/train_*_low.tif")))
    PATHS_HIGH_TRAIN_EXTRA = sorted(glob(os.path.join(cfg.DATA_ROOT, "nerima_dataset/train_*_high.tif")))
    PATHS_LOW_TRAIN_EXTRA = sorted(glob(os.path.join(cfg.DATA_ROOT, "nerima_dataset/train_*_low.tif")))

    df = pd.DataFrame(
        {
            "path_high": PATHS_HIGH_TRAIN + PATHS_HIGH_TRAIN_EXTRA,
            "path_low": PATHS_LOW_TRAIN + PATHS_LOW_TRAIN_EXTRA,
        }
    )
    
    df = pd.merge(
        df,
        pd.read_csv(os.path.join(cfg.DATA_ROOT, "train.csv"))[["path_high", "fold"]],
        how="left",
        on="path_high",
    )

    df["fold"] = df["fold"].fillna(-1).astype(int)

    # #display(df.head(6))

    df.to_csv(os.path.join(cfg.DATA_ROOT, "train_nerima.csv"), index=False)

    #######################################
    # load train df
    #######################################
    df = pd.read_csv(cfg.DATA_ROOT / "train_nerima.csv")
    if cfg.debug:
        tmp_list: list[pd.DataFrame] = []
        for fold, df_fold in df.groupby("fold"):
            tmp_list.append(df_fold.sample(cfg.debug_sample // cfg.folds))
        df = pd.concat(tmp_list)

    #display(df)
    #display(df.shape)

    #######################################
    # load test df
    #######################################

    test_df = pd.DataFrame({"path_low": natsorted(cfg.ROOT_TEST.glob("test_*_low.tif"))}) # type: ignore
    test_df = test_df.assign(path_low=test_df["path_low"].apply(lambda x: x.as_posix()))
    #display(test_df.info())
    if cfg.debug:
        test_df = test_df.sample(cfg.debug_sample, random_state=cfg.seed)

    #######################################
    # Augmentation
    #######################################
    # augmentation
    tf_dict = {"train": A.Compose([A.Transpose(),
                                A.Flip(),
                                A.RandomRotate90(),
                                ToTensorV2()], is_check_shapes=False),
                "train_augless": A.Compose([ToTensorV2()], is_check_shapes=False),
                "val": A.Compose([ToTensorV2()], is_check_shapes=False)}

    tf_dict["test"] = tf_dict["val"]

    cfg.augmentation = str(tf_dict).replace("\n", "").replace(" ", "")
    #display(cfg.augmentation)

    #######################################
    # train for each fold
    #######################################
    for fold in cfg.trn_folds:
        set_seed(cfg.seed + fold)
        print("■" * 30, f"fold: {fold}", "■" * 30)

        # train val split
        train_df = df[df["fold"] != fold].reset_index(drop=True)
        val_df = df[df["fold"] == fold].reset_index(drop=True)
        print(f"[num sample] train: {len(train_df)} val:{len(val_df)}")
        assert (
            len(train_df) > 0 and len(val_df) > 0
        ), f"[Num Sample] train: {len(train_df)} val:{len(val_df)}"

        datamodule = CustomDataLoaderModule(df_train=train_df, df_val=val_df, cfg=cfg, tf_dict=tf_dict)
        model = SRModel(cfg, fold)

        ckpt_name = cfg.ckpt_stem + f"_fold{fold}"
        if delete_if_file_exists(cfg.outdir / f"{ckpt_name}.ckpt"):
            print(f"delete existing ckpt: {ckpt_name}.ckpt")

        # metrics
        save_checkpoint_callback = callbacks.ModelCheckpoint(  # ここでモデル保存してるのか
            dirpath=cfg.outdir,
            filename=ckpt_name,
            monitor="val/loss",
            save_top_k=1,
            mode="min",
            save_last=False,
        )

        # logger
        logging_dir = f"output/{cfg.group}/{cfg.runname}_fold{fold}"
        tb_logger = TensorBoardLogger(logging_dir)
        lr_monitor = callbacks.LearningRateMonitor()
        callbacks_list = [lr_monitor, save_checkpoint_callback]

        # trainer
        trainer = Trainer(
            logger=[tb_logger],
            max_epochs=cfg.epoch,
            callbacks=callbacks_list,
            **cfg.trainer,
        )
        trainer.fit(model, datamodule=datamodule)
        print(Fore.CYAN + f"fold:{fold} validation score:", trainer.model.score_max) # type: ignore

        del model, trainer, datamodule
        torch.cuda.empty_cache()
        # inference for test
        test_ds = CustomDataset(df=test_df, phase="test", tf_dict=tf_dict, cfg=cfg)
        test_dl = DataLoader(test_ds, batch_size=cfg.val_loader.batch_size, shuffle=False)
        model = SRModel.load_from_checkpoint(
            cfg.outdir / (cfg.ckpt_stem + f"_fold{fold}.ckpt"), cfg=cfg, fold=fold
        )
        inferece_fn(model, test_dl, cfg.outdir / f"test_fold{fold}", cfg=cfg)

        del model, test_ds, test_dl
        torch.cuda.empty_cache()
        gc.collect()

    score_each_fold: dict[int, float] = defaultdict(lambda: None)  # type: ignore # foldごとのスコアを保存する辞書

    oof_file_paths = []
    oof_directories = sorted(cfg.outdir.glob("oof_data_fold*"))
    for oof_directory in oof_directories:
        # oof_directory: Path = cfg.outdir / f"oof_data_fold{fold}"
        fold = int(oof_directory.name.split("fold")[-1])
        oof_file_paths_fold = sorted(oof_directory.glob("train_*_oof.tif"))
        score_each_fold[fold] = valid_fn(oof_file_paths=oof_file_paths_fold, cfg=cfg)
        oof_file_paths.extend(oof_file_paths_fold)

    score_each_fold[-1] = valid_fn(oof_file_paths=oof_file_paths, cfg=cfg)
    print(Fore.CYAN + f"Overall CV: {score_each_fold[-1]:.6f}")

    # show scores
    print("■" * 10 + "result" + "■" * 10)
    score_str = []
    for k in [-1] + list(range(cfg.folds)):
        v = score_each_fold[k]
        if k == -1:
            k = " all"
        if v is None:
            score_str.append(" ")
        else:
            score_str.append(f"{v:.6f}")
            print(f"fold{k}: {v:.6f}")

    print(", ".join(score_str))
    # スプレッドシートにコピペできるように

    merge_inference(cfg.outdir)


def validate_gpus_argument(argument:str) -> Union[List, int]:
    try:
        if "," in argument:
            argument = [int(x) for x in argument.split(",")] # type: ignore # Convert to list of integers
        else:
            argument = int(argument) # type: ignore # Convert to single integer
    except:
        traceback.print_exc()

    return argument # type: ignore

def validate_strategy_argument(argument:str) -> Union[str, bool]:
    if argument.lower() == "false":
        argument = False # type: ignore # Convert to boolean False
    elif argument.lower() == "ddp":
        argument = "ddp"
    else:
        raise ValueError('Training strategy not defined correctly, choose between "ddp" or False')

    return argument

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using the given dataset.")
    parser.add_argument("--own_dataset", type=bool, default=False, help="Use own dataset instead of competition dataset.")
    parser.add_argument("--use_gpus", type=str, default="1", help="Number of GPUs to use for training. Can be a comma-separated list of GPU IDs or a single integer.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode for training.")
    parser.add_argument("--strategy", type=str, default=False, help='Training strategy, choose between "ddp" or False.')
    
    args = parser.parse_args()
    using_own_dataset = args.own_dataset
    gpus = validate_gpus_argument(args.use_gpus)
    debug = args.debug
    strategy = validate_strategy_argument(args.strategy)

    if isinstance(gpus, int):
        gpus = [gpus]

    train(using_own_dataset=using_own_dataset, gpus=gpus, debug=debug, strategy=strategy)