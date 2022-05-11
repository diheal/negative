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
+ main.py: This file contains the details of ClusterClHAR and the methods how it works.

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
