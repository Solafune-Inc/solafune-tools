"""from train_src.train_step1 import train as train_step1
from train_src.train_step2 import train as train_step2
from train_src.train_step3 import train as train_step3
from train_src.train_step4 import train as train_step4
from train_src.train_continue import train as train_continue"""
from train_src.train_end import move_weights
import torch
import traceback
import argparse
from typing import Union, List, Tuple
import os
import sys

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

def validate_continue_training(argument:str) -> Tuple[bool, str]:
    if argument.lower() == "false":
        mid_training = False
        models_dir = ""
    else:
        mid_training = True
        models_dir = argument
    return mid_training, models_dir

def main(gpus: Union[List, int], strategy: Union[bool, str] = False, using_own_dataset:bool = False, debug: bool = False, continue_training: Tuple[bool, str] = (False, "")):
    """
    Executes the main function.

    This function calls the following steps in sequence:
    - `train_step1()`: Executes the first step of the training process.
    - `torch.cuda.empty_cache()`: Clears the CUDA memory cache.
    - `train_step2()`: Executes the second step of the training process.
    - `torch.cuda.empty_cache()`: Clears the CUDA memory cache.
    - `train_step3()`: Executes the third step of the training process.
    - `torch.cuda.empty_cache()`: Clears the CUDA memory cache.
    - `train_step4()`: Executes the fourth step of the training process.
    - `torch.cuda.empty_cache()`: Clears the CUDA memory cache.
    """
    mid_training, models_dir = continue_training

    """if mid_training:
        train_continue(using_own_dataset=using_own_dataset, gpus=gpus, debug = debug, strategy = strategy, models_dir=models_dir)
        exit()

    train_step1(using_own_dataset=using_own_dataset, gpus=gpus, debug = debug, strategy = strategy)
    torch.cuda.empty_cache()
    train_step2(using_own_dataset=using_own_dataset, gpus=gpus, debug = debug, strategy = strategy)
    torch.cuda.empty_cache()
    train_step3(using_own_dataset=using_own_dataset, gpus=gpus, strategy = strategy)
    torch.cuda.empty_cache()
    train_step4(using_own_dataset=using_own_dataset, gpus=gpus, debug = debug, strategy = strategy)
    torch.cuda.empty_cache()
    move_weights()"""

    # Alternative using shell script
    args = f"--use_gpus {gpus} --strategy {strategy} {'--debug' if debug else ''} {'--own_dataset' if using_own_dataset else ''}"
    if mid_training:
        args += f" --continue-training {models_dir}"
        os.system(f"python train_src/train_continue.py {args}")

    else:
        print(f"Argurments for training: {args}")
        print("-- Starting training step 1 --")
        os.system(f"python -m train_src.train_step1 {args}")
        torch.cuda.empty_cache()
        print("-- Starting training step 2 --")
        os.system(f"python -m train_src.train_step2 {args}")
        torch.cuda.empty_cache()
        print("-- Starting training step 3 --")
        os.system(f"python -m train_src.train_step3 {args}")
        torch.cuda.empty_cache()
        print("-- Starting training step 4 --")
        os.system(f"python -m train_src.train_step4 {args}")
        torch.cuda.empty_cache()
    print("-- Moving weights --")
    move_weights()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Train Super Resolution model using competition dataset or your own dataset")
    parser.add_argument("--own_dataset", action="store_true", default=False)
    parser.add_argument("--use_gpus", type=str, default="1")
    parser.add_argument("--strategy", type=str, default="False")
    parser.add_argument("--continue-training", type=str, default="False")
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()
    main(using_own_dataset=args.own_dataset, gpus=args.use_gpus, 
         strategy=args.strategy, continue_training=validate_continue_training(args.continue_training), 
         debug = args.debug)