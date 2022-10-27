# M<sup>3</sup>VSNet
The code is available now!!!
![M3VSNet](https://github.com/whubaichuan/M3VSNet/blob/master/img/network.png)


## About
The present Multi-view stereo (MVS) methods with supervised learning-based networks have an impressive performance comparing with traditional MVS methods. However, the ground-truth depth maps for training are hard to be obtained and are within limited kinds of scenarios. In this paper, we propose a novel unsupervised multi-metric MVS network, named M<sup>3</sup>VSNet, for dense point cloud reconstruction without any supervision. To improve the robustness and completeness of point cloud reconstruction, we propose a novel multi-metric loss function that combines pixel-wise and feature-wise loss function to learn the inherent constraints from different perspectives of matching correspondences. Besides, we also incorporate the normal-depth consistency in the 3D point cloud format to improve the accuracy and continuity of the estimated depth maps. Experimental results show that M<sup>3</sup>VSNet establishes the state-of-the-arts unsupervised method and achieves comparable performance with previous supervised MVSNet on the DTU dataset and demonstrates the powerful generalization ability on the Tanks and Temples benchmark with effective improvement.


Please cite: 
```
@inproceedings{huang2021m3vsnet,
  title={M3VSNet: Unsupervised multi-metric multi-view stereo network},
  author={Huang, Baichuan and Yi, Hongwei and Huang, Can and He, Yijia and Liu, Jingbin and Liu, Xiao},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={3163--3167},
  year={2021},
  organization={IEEE}
}
```

## How to use
### Environment
- python 3.6.9
- pytorch 1.0.1
- CUDA 10.1 cudnn 7.5.0

The conda environment is listed in [requirements.txt](https://github.com/whubaichuan/M3VSNet/blob/master/requirements.txt)

### Train
* Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) (Fixed training cameras, from [Original MVSNet](https://github.com/YoYo000/MVSNet)，or the Baiduyun [link](https://pan.baidu.com/s/1sQAC3pmceyochNvnqpE9oA), the password is mo8w ), and upzip it as the ``MVS_TRANING`` folder
* in ``train.sh``, set ``MVS_TRAINING`` as your training data path
* create a logdir called ``checkpoints``
* Train MVSNet: ``./train.sh``

### Eval
* Download the preprocessed test data [DTU testing data](https://drive.google.com/open?id=135oKPefcPTsdtLRzoDAQtPpHuoIrpRI_) (from [Original MVSNet](https://github.com/YoYo000/MVSNet), or the Baiduyun [link](https://pan.baidu.com/s/1sQAC3pmceyochNvnqpE9oA), the password is mo8w ) and unzip it as the ``DTU_TESTING`` folder, which should contain one ``cams`` folder, one ``images`` folder and one ``pair.txt`` file.
* in ``test.sh``, set ``DTU_TESTING`` as your testing data path and ``CKPT_FILE`` as your checkpoint file. You can find some models in the /checkpoint/. You can use the trained models to test your image.
* Test MVSNet: ``./test.sh``

## Results
![Results](https://github.com/whubaichuan/M3VSNet/blob/master/img/result.png)

|                       | Acc.   | Comp.  | Overall. |
|-----------------------|--------|--------|----------|
| MVSNet(D=196)         | 0.444  | 0.741  | 0.592    |
| Unsup_MVS         | 0.881  | 1.073  | 0.977    |
| MVS2         | 0.760  | 0.515  | 0.637   |
| M3VSNet(D=192) | 0.636 | 0.531 | 0.583   |

### T&T Benchmark
The best unsupervised MVS network until April 17, 2020. See the [leaderboard ](https://www.tanksandtemples.org/details/853/). 

## Acknowledgement
Thanks for the funding from [Megvii Technology Limited](https://www.megvii.com/). We acknowledge the following repositories [MVSNet](https://github.com/YoYo000/MVSNet) and [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch). 

Happy to be acknowledgemented by the AAAI 2020 [paper](https://github.com/ToughStoneX/Self-Supervised-MVS).  
