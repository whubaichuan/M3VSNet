# M<sup>3</sup>VSNet
Code openning soon！

## About
The present Multi-view stereo (MVS) methods with supervised learning-based networks have an impressive performance comparing with traditional MVS methods. However, the ground-truth depth maps for training are hard to be obtained and are within limited kinds of scenarios. In this paper, we propose a novel unsupervised multi-metric MVS network, named M<sup>3</sup>VSNet, for dense point cloud reconstruction without any supervision. To improve the robustness and completeness of point cloud reconstruction, we propose a novel multi-metric loss function that combines pixel-wise and feature-wise loss function to learn the inherent constraints from different perspectives of matching correspondences. Besides, we also incorporate the normal-depth consistency in the 3D point cloud format to improve the accuracy and continuity of the estimated depth maps. Experimental results show that M<sup>3</sup>VSNet establishes the state-of-the-arts unsupervised method and achieves comparable performance with previous supervised MVSNet on the DTU dataset and demonstrates the powerful generalization ability on the Tanks and Temples benchmark with effective improvement.


Please cite: 
```
@article{Huang2020M3VSNet,
  title={M<sup>3</sup>VSNet: Unsupervised Multi-metric Multi-view Stereo Network},
  author={Baichuan Huang and Hongwei Yi and Can Huang and Yijia He and Jingbin Liu and Xin Liu},
  journal={ArXiv},
  year={2020},
  volume={abs/2004.09722v2}
}
```

## How to use
### Environment
- python 3.6.9
- pytorch 1.0.1
- CUDA 10.1 cudnn 7.5.0

The conda environment is listed in [requirements.txt](https://github.com/whubaichuan/M3VSNet/blob/master/requirements.txt)

### Train

### Eval

## Results

### T&T Benchmark
The best unsupervised MVS network until April 17, 2020. See the [leaderboard ](https://www.tanksandtemples.org/details/853/). 

## Acknowledgement
We acknowledge the following repositories [MVSNet](https://github.com/YoYo000/MVSNet) and [MVSNet_pytorch](https://github.com/xy-guo/MVSNet_pytorch). For more information about MVSNet series, please see the [material](https://mp.weixin.qq.com/s/fnKU4dkYvBEU913Vanj54Q).
