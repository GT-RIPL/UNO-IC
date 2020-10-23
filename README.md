# Imbalaced Classification and Robust Semantic Segmentation 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repo implements two algoritms.
1. The imbalance clibration algorithm for image classification.
2. The uncertainty-aware noisy-or multi-modal fusion and imbalance calibration algorithm for semantic segmentation, UNO-IC. 

## Imbalanced Image Classification
```
git checkout classification
```
## Robust Multi-Modal Fusion for Semantic Segmentation
```
git checkout segmentation
```


## Acknowledgments
- This work was supported by ONR grant N00014-18-1-2829.
- This code is built upon the implementation from [Pytorch-semseg](https://github.com/meetshah1995/pytorch-semseg).

## Citation
If you find this repository useful, please cite our paper(s):

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
