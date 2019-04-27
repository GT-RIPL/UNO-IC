# pytorch-semseg (RIPL README)

0. Download airsim dataset

- currently stored on ripl-w3 (ripl@143.215.207.42)
  - /home/datasets/airsim_03-30-2019
- ask Yen-Chang for password
- path parameter in config file should point to ./datasets/airsim_data_async
- symbolic link ./datasets/airsim_data_async to dataset (airsim_03-30-2019/airsim_data_async)
- no trailing backslash in path variable
- folder should contain subfolders of modalities: (depth, depth_encoded, pose, scene, segmentation)
1. Install: follow original readme installation instructions
2. Adjust configuration: configs/XXX.yml file
3. Train:

- General command: CUDA_VISIBLE_DEVICES=0 python train.py
- To train RGB leg: CUDA_VISIBLE_DEVICES=0 python train.py --config configs/BayesianSegnet/rgb_BayesianSegnet_0.5_T000.yml
- To train D leg: CUDA_VISIBLE_DEVICES=0 python train.py --config configs/BayesianSegnet/d_BayesianSegnet_0.5_T000.yml
- To train/test SoftmaxFusion: CUDA_VISIBLE_DEVICES=0 python train.py --config configs/BayesianSegnet/outputFusion_FusionSoftmaxMultiply_Train000_Recal{None,000,050,100}.yml

4. Tensorboard progress monitor: run tensorboard from within the "runs" directory

- How to read?
- Set pagination limit in tensorboard to match number of plots for a given split
- First scrollable page shows training and validation losses
- Second scrollable page shows accuracies
- Title of plots describes the validation split (i.e. async_fog_000_clear_...)

5. Compare final results: tools/tensorboard_segnet.py

- This file generates a bar chart comparing the final data points from the training run
- This file needs to be symlinked into a folder that contains the runs you want to compare
  - the first level should have folders that group similar tests together
  - the second level should have the folders produced by the training code (these folders themselves should have the tensorboard log files)
- Things to change in the file itself
  - include = ... (should have a list of of those previously created grouping folders, you can comment out specific groupings here if you don't want their results in the final barchart)
  - run_comments
    - keys in dictionary should be the folder names listed in "include"
      - "names" should have a list of tuples
        - the left element should be the name of the folder produced by the training code
        - the right element should be a pretty name that describes the specific test
      - "text" should be your comments about the grouping of tests



Notable Things:
- configuration files are stored in ./configs directory
- tensorboard files, checkpoint weights, log files, and corresponding run config files are stored in ./runs directory
- almost all code is contained in train.py, ./ptsemseg/models/{semseg_mcdo.py,recalibrator.py} files


Fusion Specific Configuration File:

```yaml
# Model Configuration
id: <pretty name for the current test>

model:
    arch: <name> [options: 'segnet_mcdo, fcn[8,16,32]s, unet, segnet, pspnet, icnet, icnetBN, linknet, frrn[A,B]'
    <model_keyarg_1>:<value>

models: # list of model dicts that will be spawned
    rgb: (or d)
        in_channels: 3              #number of input channels
        start_layer: down1          #start layer (inclusive) of architecture (used for mid mcdo fusion)
        end_layer: up1              #final layer (inclusive) of architecture (used for mid mcdo fusion)
        reduction: 1.0              #not implemented (used for reducing the number of parameters in the network)
        mcdo_passes: 1              #number of passes for MCDO
        dropoutP: 0.0               #dropout probability for MCDO
        mcdo_start_iter: 0          #not implemented
        mcdo_backprop: False        #not implemented
        learned_uncertainty: "no"   #not implemented
        resume: None                #path to the .pkl checkpoint file


# Data Configuration
data:
    dataset: <name> [options: 'airsim, pascal, camvid, ade20k, mit_sceneparsing_benchmark, cityscapes, nyuv2, sunrgbd, vistas'] 
    train_split: <split_to_train_on>
    train_subsplit: ['<specific_environment_folder_to_train_on>']
    train_reduction: 1.0 #fractional reduction of train set size
    val_split: <spit_to_validate_on>
    val_subsplit: ['<specific_environment_folder_to_validate_on>']
    val_reduction: 1.0 #fractional reduction of validation set size
    img_rows: 512
    img_cols: 512
    path: <path/to/data>
    <dataset_keyarg1>:<value>

# Training Configuration
training:
    n_workers: 64
    train_iters: 35000
    batch_size: 16
    val_interval: 500
    print_interval: 25
    loss:
        name: <loss_type> [options: 'cross_entropy, bootstrapped_cross_entropy, multi_scale_crossentropy']
        <loss_keyarg1>:<value>

    # Optmizer Configuration
    optimizer:
        name: <optimizer_name> [options: 'sgd, adam, adamax, asgd, adadelta, adagrad, rmsprop']
        lr: 1.0e-3
        <optimizer_keyarg1>:<value>

        # Warmup LR Configuration
        warmup_iters: <iters for lr warmup>
        mode: <'constant' or 'linear' for warmup'>
        gamma: <gamma for warm up>
       
    # Augmentations Configuration
    augmentations:
        gamma: x                                     #[gamma varied in 1 to 1+x]
        hue: x                                       #[hue varied in -x to x]
        brightness: x                                #[brightness varied in 1-x to 1+x]
        saturation: x                                #[saturation varied in 1-x to 1+x]
        contrast: x                                  #[contrast varied in 1-x to 1+x]
        rcrop: [h, w]                                #[crop of size (h,w)]
        translate: [dh, dw]                          #[reflective translation by (dh, dw)]
        rotate: d                                    #[rotate -d to d degrees]
        scale: [h,w]                                 #[scale to size (h,w)]
        ccrop: [h,w]                                 #[center crop of (h,w)]
        hflip: p                                     #[flip horizontally with chance p]
        vflip: p                                     #[flip vertically with chance p]

    # LR Schedule Configuration
    lr_schedule:
        name: <schedule_type> [options: 'constant_lr, poly_lr, multi_step, cosine_annealing, exp_lr']
        <scheduler_keyarg1>:<value>

    # Resume from checkpoint  
    resume: <path_to_checkpoint>
```


# pytorch-semseg (Original README)

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/meetshah1995/pytorch-semseg/blob/master/LICENSE)
[![pypi](https://img.shields.io/pypi/v/pytorch_semseg.svg)](https://pypi.python.org/pypi/pytorch-semseg/0.1.2)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1185075.svg)](https://doi.org/10.5281/zenodo.1185075)



## Semantic Segmentation Algorithms Implemented in PyTorch

This repository aims at mirroring popular semantic segmentation architectures in PyTorch. 


<p align="center">
<a href="https://www.youtube.com/watch?v=iXh9aCK3ubs" target="_blank"><img src="https://i.imgur.com/agvJOPF.gif" width="364"/></a>
<img src="https://meetshah1995.github.io/images/blog/ss/ptsemseg.png" width="49%"/>
</p>

### Networks implemented

* [PSPNet](https://arxiv.org/abs/1612.01105) - With support for loading pretrained models w/o caffe dependency
* [ICNet](https://arxiv.org/pdf/1704.08545.pdf) - With optional batchnorm and pretrained models
* [FRRN](https://arxiv.org/abs/1611.08323) - Model A and B
* [FCN](https://arxiv.org/abs/1411.4038) - All 1 (FCN32s), 2 (FCN16s) and 3 (FCN8s) stream variants
* [U-Net](https://arxiv.org/abs/1505.04597) - With optional deconvolution and batchnorm
* [Link-Net](https://codeac29.github.io/projects/linknet/) - With multiple resnet backends
* [Segnet](https://arxiv.org/abs/1511.00561) - With Unpooling using Maxpool indices


### DataLoaders implemented

* [AirSim](https://github.com/Microsoft/AirSim)
* [CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)
* [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html)
* [ADE20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/)
* [MIT Scene Parsing Benchmark](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)
* [Cityscapes](https://www.cityscapes-dataset.com/)
* [NYUDv2](http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
* [Sun-RGBD](http://rgbd.cs.princeton.edu/)


### Requirements

* pytorch >=0.4.0
* torchvision ==0.2.0
* scipy
* tqdm
* tensorboardX

#### One-line installation
    
`pip install -r requirements.txt`

### Data

* Download data for desired dataset(s) from list of URLs [here](https://meetshah1995.github.io/semantic-segmentation/deep-learning/pytorch/visdom/2017/06/01/semantic-segmentation-over-the-years.html#sec_datasets).
* Extract the zip / tar and modify the path appropriately in your `config.yaml`


### Usage

**Setup config file**

```yaml
# Model Configuration
model:
    arch: <name> [options: 'fcn[8,16,32]s, unet, segnet, pspnet, icnet, icnetBN, linknet, frrn[A,B]'
    <model_keyarg_1>:<value>

# Data Configuration
data:
    dataset: <name> [options: 'pascal, camvid, ade20k, mit_sceneparsing_benchmark, cityscapes, nyuv2, sunrgbd, vistas'] 
    train_split: <split_to_train_on>
    val_split: <spit_to_validate_on>
    img_rows: 512
    img_cols: 1024
    path: <path/to/data>
    <dataset_keyarg1>:<value>

# Training Configuration
training:
    n_workers: 64
    train_iters: 35000
    batch_size: 16
    val_interval: 500
    print_interval: 25
    loss:
        name: <loss_type> [options: 'cross_entropy, bootstrapped_cross_entropy, multi_scale_crossentropy']
        <loss_keyarg1>:<value>

    # Optmizer Configuration
    optimizer:
        name: <optimizer_name> [options: 'sgd, adam, adamax, asgd, adadelta, adagrad, rmsprop']
        lr: 1.0e-3
        <optimizer_keyarg1>:<value>

        # Warmup LR Configuration
        warmup_iters: <iters for lr warmup>
        mode: <'constant' or 'linear' for warmup'>
        gamma: <gamma for warm up>
       
    # Augmentations Configuration
    augmentations:
        gamma: x                                     #[gamma varied in 1 to 1+x]
        hue: x                                       #[hue varied in -x to x]
        brightness: x                                #[brightness varied in 1-x to 1+x]
        saturation: x                                #[saturation varied in 1-x to 1+x]
        contrast: x                                  #[contrast varied in 1-x to 1+x]
        rcrop: [h, w]                                #[crop of size (h,w)]
        translate: [dh, dw]                          #[reflective translation by (dh, dw)]
        rotate: d                                    #[rotate -d to d degrees]
        scale: [h,w]                                 #[scale to size (h,w)]
        ccrop: [h,w]                                 #[center crop of (h,w)]
        hflip: p                                     #[flip horizontally with chance p]
        vflip: p                                     #[flip vertically with chance p]

    # LR Schedule Configuration
    lr_schedule:
        name: <schedule_type> [options: 'constant_lr, poly_lr, multi_step, cosine_annealing, exp_lr']
        <scheduler_keyarg1>:<value>

    # Resume from checkpoint  
    resume: <path_to_checkpoint>
```

**To train the model :**

```
python train.py [-h] [--config [CONFIG]] 

--config                Configuration file to use
```

**To validate the model :**

```
usage: validate.py [-h] [--config [CONFIG]] [--model_path [MODEL_PATH]]
                       [--eval_flip] [--measure_time]

  --config              Config file to be used
  --model_path          Path to the saved model
  --eval_flip           Enable evaluation with flipped image | True by default
  --measure_time        Enable evaluation with time (fps) measurement | True
                        by default
```

**To test the model w.r.t. a dataset on custom images(s):**

```
python test.py [-h] [--model_path [MODEL_PATH]] [--dataset [DATASET]]
               [--dcrf [DCRF]] [--img_path [IMG_PATH]] [--out_path [OUT_PATH]]
 
  --model_path          Path to the saved model
  --dataset             Dataset to use ['pascal, camvid, ade20k etc']
  --dcrf                Enable DenseCRF based post-processing
  --img_path            Path of the input image
  --out_path            Path of the output segmap
```


**If you find this code useful in your research, please consider citing:**

```
@article{mshahsemseg,
    Author = {Meet P Shah},
    Title = {Semantic Segmentation Architectures Implemented in PyTorch.},
    Journal = {https://github.com/meetshah1995/pytorch-semseg},
    Year = {2017}
}
```

