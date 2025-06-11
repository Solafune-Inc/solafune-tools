# 5x Super Resolution
## Model Card Description
This write-up is for model inference and model training of Super Resolution released by Solafune. This x5 super resolution model is based on Team N's solution, the first place in the competition of [5x Super Resolution Competition](https://solafune.com/competitions/7a1fc5e3-49bd-4ec1-8378-974951398c98?menu=about&tab=overview). The model was based on SWIN2SR and developed into 5x super resolution.
## Usage
To run the model inference and training without any constraint, this step must be followed thoroughly
1. Installing the Requirements
   - If directly from pip or uv
     ```bash
     pip install solafune-tools
     pip install solafune-tools[super-resolution]
     ```
     
   - If directly from the library
     ```bash
     pip install .
     pip install .[super-resolution]
     ```

2. Running the inference modules
  ### Module Import
   To run the inference using python `solafune_tools` library import, you can use this usage tutorial to help you first time using this 5x Super Resolution model inference. Please prepare the first image you want the inference first. JPG/JPEG, PNG, TIF, and TIFF are the acceptable file extensions but basically as long as the format is numpyArray also acceptable. This example lines of code will let you run the model inference.
  ```python
  from solafune_tools.osm.super_resolution.inference import Model
  import tifffile
  
  SR_Inference = Model()

  img_array = tifffile.TiffFile(img_input).asarray() # Make sure the input is in RGB bands
  img_result = SR_Inference.generate(img_array)

  tifffile.imwrite("result.tif", img_result)
  ```
  ### Bash/CMD panel Interface
   To run the inference through bash/cmd panel interface, please prepare the first image you want the inference first. JPG/JPEG, PNG, TIF, and TIFF are the acceptable file extensions. This bash example will let you run the model inference.
   ```bash
   python inference.py --input sample_input/mango.png # This will lead to the default output saved in the output/output.tif
   or
   python inference.py --input sample_input/mango.png --output example.jpg # The output will be written in the current directory with the desired name and extension.
   ```

3. Running the model training
   > You can use your dataset or the competition dataset
   > There are extras dataset that is part of the code model training.
   - When you want to use the competition dataset, you don't need to download it first. The program inside the dataset module will automatically download from the S3 bucket storage and put them structurally comply with how the training will be commenced. To train the model you only need to use this bash script:
     ```bash
     python train.py
     ```
   - If you want to test whether the training will run or not, you can use `--debug` flag to find out if something is wrong with your environment without taking the time to wait for all epochs
     ```bash
     python train.py --debug
     ```
   - If you want to use multi-GPU and Distributed Data-Parallel(DDP), you can use the argument flag `--use_gpus 0,1,2,3... or 1/2/4` and `--strategy ddp` to choose how many GPUs you want to use and the use of DDP strategy in training. The default is using one GPU and not using DDP.
     ```bash
     python train.py --use_gpus 0,1,2,4 --strategy ddp # To use GPUs number 1, 2, 3, and 5 and using the ddp strategy.  
     or
     python train.py --use_gpus 4 --strategy ddp # To use 4 GPUs if you have and using the ddp strategy. 
     ```
   - If you want to use your dataset, please comply with the dataset structure tree. Failure to comply will result in training being aborted
     ```tree
       datasets
          ├── test
          │   ├── *_low.tif 
          │   └── ... 
          └── train
              ├── *_high.tif
              ├── *_low.tif
              └── ...
     ```
     After you comply with the dataset structure, you can run the script below to set the True option using your dataset
     ```bash
     python train.py --own_dataset
     ```
     Because there are about 4 steps in training the model, it will take around 10-15 days to train all the models, depending on how many main datasets or your type of GPUs.
   - If you want to continue your last training after you have done your first model training, you can use this flag `--continue-training` followed by your models' directory where you keep the trained model checkpoint. Please follow this naming format for your mid-training, like this folder tree Also follow the four-fold arrangement starting from number zero to number three.
      ```tree
      working
         ├── trained_model_fold0.ckpt
         ├── trained_model_fold1.ckpt
         ├── trained_model_fold2.ckpt
         └── trained_model_fold3.ckpt
      ```
      Here is the following CLI example for mid-training continuing.
     ```bash
     python train.py --own_dataset --use_gpus 0,3,5,7 --strategy ddp --continue-training working # 'working' indicated where model's directory are
     ```
   - Full example to test if all functions are working as intended
     ```bash
     python train.py --own_dataset --use_gpus 0,3,5,7 --strategy ddp --debug
     ```
        
