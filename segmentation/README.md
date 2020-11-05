# UNO-IC (This repo implements the uncertainty-aware noisy-or fusion and imbalance calibration algorithm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 


## Download Synthia dataset

- The model uses the SEQS-05 Sythia-seq collections [here](http://synthia-dataset.net/downloads/)
- Using different collections might require modifications to the dataloader. Please check the specific data structure and labels.
- Extract the zip / tar and modify the path appropriately in your `./configs/eval/rgbd_synthia.yml`
```yaml
path: /datasets/synthia-seq/
```

## Create conda environment
```
conda env create -f requirements.yaml
conda activate uno-ic
```

## Pre-trained Models 
- We train the model using four conditions. 
```yaml
data:
    dataset: synthia
    train_split: train
    train_subsplit: ['SYNTHIA-SEQS-05-DAWN',
                   'SYNTHIA-SEQS-05-SUMMER',
                   'SYNTHIA-SEQS-05-NIGHT', 
                   'SYNTHIA-SEQS-05-SUNSET',]
```
- Pretrained RGB and D models are provided [here](https://drive.google.com/file/d/1tlHa0PF5nK0SS1gPuTCKBz1yC4Q_ARIS/view?usp=sharing)
- Download and save the pretrained models to `./checkpoint/synthia-seq/deeplab/` 
```
$ mkdir -p ./checkpoint/synthia-seq/deeplab/
```


## Inference
1. Extract training entropy statistics for uncertainty scaling and label priors for imbalance calibration.
- Assign `val` to `val_split` in the evaluation configuration file (`./configs/eval/rgbd_synthia.yml`)
```yaml
val_split: val
    val_subsplit: [
                  'SYNTHIA-SEQS-05-DAWN',
                  'SYNTHIA-SEQS-05-SUMMER',
                  'SYNTHIA-SEQS-05-NIGHT', 
                  'SYNTHIA-SEQS-05-SUNSET',]
```
- Run `python extract.py --config ./configs/synthia/eval/rgbd_synthia.yml`

- Statistics will be saved in foler `./runs/sythia/stats`

2. Run inference
- Assign  `test` to `val_split` and comment/uncomment attributes of `val_subsplit` to test on your selected conditions.
```yaml
val_split: test 
```
- Run `python validate.py --config ./configs/synthia/eval/rgbd_synthia.yml` 
- Qulitative and quatitative results will be saved in `runs/synthia/rgbd_synthia`

## Apply additional degradations
- We apply the photorealitic degradations from this [repo](https://github.com/hendrycks/robustness)
- Please make the following modification in the configuration file to apply additional degradations.
```
SYNTHIA-SEQS-05-DAWN__{'channel':'rgb','type':'defocusBlur','value':'3'}"
```
- Applicable degradation type: 
```
gaussianNoise, shotNoise, impulseNoise, defocusBlur, glassBlur, motionBlur
zoomBlur, snow, frost, fog, brightness, contrast, elastic, pixelate
```
- Degradation value ranges from 1 to 5

## Key functions
- Uncertainty Scaling: assign `True` to `uncertainty:` in the evaluation configuration file
- Imblance Calibration: assign a non-negative scalar to `beta:` to use imbalance calibration else leave it blank in the evaluation configuration file
- Fusion: assign any valid fusion stretagy among `Noisy-Or, SoftmaxAverage, SoftmaxMultiply` to `fusion:` in the evaluation configuration file


## Acknowledgments
- This work was supported by ONR grant N00014-18-1-2829.
- This code is built upon the implementation from [Pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg).

## Citation
If you find this repository useful, please cite our paper:

```
@inproceedings{tian2019uno,
    title={UNO: Uncertainty-aware Noisy-Or Multimodal Fusion for Unanticipated Input Degradation},
    author={Junjiao Tian, Wesley Cheung,  Nathaniel Glaser, Yen-Cheng Liu, and Zsolt Kira},
    booktitle={Proceedings of the  IEEE International Conference on Robotics and Automation (ICRA)},
    year={2020}
}

@inproceedings{tian2020posterior,
    title={Posterior Re-calibration for Imbalanced Datasets},
    author={Junjiao Tian, Yen-Cheng Liu,  Nathaniel Glaser, Yen-Chang Hsu, and Zsolt Kira},
    booktitle={Proceedings of the 34th Conference on Neural Information Processing Systems},
    year={2020}
}
```
