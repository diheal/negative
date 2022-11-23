# Negative Selection by Clustering for Contrastive Learning in Human Activity Recognition
This repository is designed to implement the idea of "Negative Selection by Clustering for Contrastive Learning in Human Activity Recognition" in this paper.
For more details, please see https://arxiv.org/abs/2203.12230
## Requirements

This project code is done in Python 3.8 and third party libraries. 

 TensorFlow 2.x is used as a deep learning framework.

The main third-party libraries used and the corresponding versions are as follows:

+ tensorflow 2.3.1

+ tensorflow_addons 0.15.0

+ numpy 1.18.5

+ scipy 1.5.0

+ scikit-learn 0.23.1


  

## Running

This demo can be run with the following command:

```shell
python main.py
```


## Code Organisation

The main content of each file is marked as follows:

+ Augment.py: This file contains a variety of sensor data augmentation methods.
+ TPN.py: This file contains the network structure of TPN
+ main.py: This file contains the details of ClusterCLHAR and the methods how it works.

## Experiment

Results of different divisions in self-supervised experiments.

MotionSense:

| Test subjects  | Train/Val subjects  | F1-score|
|  :----:  | :----:  |:----:  |
| 1 - 5  | rest | 95.16|
| 6 - 10  | rest | 90.29|
| 11 - 15  | rest | 86.76|
| 16 - 19  | rest | 89.13|
| 20 - 24  | rest | 90.95|
| 1, 9,12, 17, 19| rest | 88.99|
| 2, 6, 10, 14, 22 | rest | 90.23|
| 3, 4, 5, 13, 16 | rest | 94.26|
| 8, 18, 21, 23, 24 | rest | 87.28|
| 11, 15, 20, 7| rest | 93.98|

UCI-HAR:

| Test subjects  | Train/Val subjects  | F1-score|
|  :----:  | :----:  |:----:  |
| 1 - 6  | rest | 94.00|
| 7 - 12  | rest | 89.59|
| 13 - 18  | rest |89.25|
| 19 - 24  | rest | 97.95|
| 25 - 30  | rest | 93.00|
| 9, 10, 16, 18, 24, 28| rest | 83.73|
| 1, 5, 13, 17, 25, 29 | rest | 95.33|
| 2, 3, 6, 12, 14, 23 | rest | 91.05|
| 4, 19, 22, 26, 27, 30 | rest | 96.61|
| 7, 8, 11, 15, 20, 21| rest | 96.45|

## Citation

If you find our paper useful or use the code available in this repository in your research, please consider citing our work:

```
@article{wang2022negative,
  title={Negative Selection by Clustering for Contrastive Learning in Human Activity Recognition},
  author={Wang, Jinqiang and Zhu, Tao and Chen, Liming and Ning, Huansheng and Wan, Yaping},
  journal={arXiv preprint arXiv:2203.12230},
  year={2022}
}
```

## Reference

+ https://github.com/iantangc/ContrastiveLearningHAR

+ https://github.com/google-research/simclr.

+ https://github.com/diheal/resampling
