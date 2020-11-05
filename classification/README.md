# Prior Rebalancing (This repo implements the imbalance calibration algorithm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

## Download Place365 dataset
- Note: This dataset is not used in the main paper. However, it is also widely used in imbalanced classification. 
- The model uses Place365 small images (256 * 256) [here](http://places2.csail.mit.edu/download.html/)
- Download the Place365-LT splits text file here [here](https://drive.google.com/file/d/14UrlzxUi12JJIX2U8NE6GFgUhsSvrUHm/view?usp=sharing). The splits are created by [liuziwei7](https://github.com/zhmiao/OpenLongTailRecognition-OLTR) 
- Extract the zip / tar and modify the path appropriately in your `./configs/place365.yml`
```yaml
data:
    name: place365
    root: /datasets/Place365/
    image_size: 224
    train:
        ann_file: /datasets/Place365/Places_LT_v2/Places_LT_train.txt
        phase: train
    val: 
        ann_file: /datasets/Place365/Places_LT_v2/Places_LT_val.txt
        phase: val
```

## Create conda environment
```
conda env create -f requirements.yaml
conda activate ic
```

## Training a CrossEntropy Baseline 
- We use 4 RTX 2080 Ti machines to the model.
```
python train.py --config ./configs/place365.yaml   
```

## Inference
1. Load trained model by specifying the path to checkpoint otherwise leave blank during training.
```yaml
resume: ../runs/place365/resnet152/CrossEntropy/ckpt.best.pth.tar  
```
2. Selecting a good beta on the validation split
- Assign `val` to `phase` in the configuration file (`./configs/place365.yml`)
```yaml
val: 
        ann_file: /datasets/Place365/Places_LT_v2/Places_LT_val.txt
        phase: val
```
- Try different `beta` under `test` and run the inference script.
```yaml
test:
    beta: 1.0  
```
```
python validate.py --config ./configs/place365.yaml   
```
3. Test on the test split using selected beta.
- Assign `test` to `phase` in the configuration file (`./configs/place365.yml`)
- Change `ann_file` to test
```yaml
val: 
        ann_file: /datasets/Place365/Places_LT_v2/Places_LT_test.txt
        phase: test
```
```
python validate.py --config ./configs/place365.yaml   
```

## Pretrained Model
- We provide a pretrained model [here](https://drive.google.com/file/d/1RNviW12oj5Dw32MxvCWHZfoe6LjlMWAl/view?usp=sharing)
- Download the pretrained model and modify the `resume` to point to the correct path.
```yaml 
resume: ../runs/place365/resnet152/CrossEntropy/ckpt.best.pth.tar 
```

## Performance
- Place365-LT test set performance

|   Method    |    [OLTR](https://arxiv.org/pdf/1904.05160.pdf)     |  [Tau-Norm](https://arxiv.org/pdf/1910.09217.pdf)   |   [LWS](https://arxiv.org/pdf/1910.09217.pdf)       |     CrossEntropy    |     Ours (1.3)    |
| :---------: | :------------: | :-----------: | :---------: | :---------: | :---------: | 
|  ResNet-152 |      35.9      |      37.9     |    37.6      |     29.86     |      **38.22**  |  

- The number in parenthesis denotes the lambda used in this experiment determined on a validation set 

## Acknowledgments
- This work was supported by ONR grant N00014-18-1-2829.
- This code is built upon the implementation from [Pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg).

## Citation
If you find this repository useful, please cite our paper(s):

```
@inproceedings{tian2020posterior,
    title={Posterior Re-calibration for Imbalanced Datasets},
    author={Junjiao Tian, Yen-Cheng Liu,  Nathaniel Glaser, Yen-Chang Hsu, and Zsolt Kira},
    booktitle={Proceedings of the 34th Conference on Neural Information Processing Systems},
    year={2020}
}
```
