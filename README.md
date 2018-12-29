# FishNet

This repo holds the implementation code of the paper:

[FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction](http://papers.nips.cc/paper/7356-fishnet-a-versatile-backbone-for-image-region-and-pixel-level-prediction.pdf)
, Shuyang Sun, Jiangmiao Pang, Jianping Shi, Shuai Yi, Wanli Ouyang, NeurIPS 2018.

FishNet was used as a key component
 for winning the 1st place in [COCO Detection Challenge 2018](http://cocodataset.org/#detection-leaderboard).

### Prerequisites
- Python 3.6.x
- PyTorch 0.4.0 (0.3.1 support soon)

### Data Augmentation

| Method | Settings |
| -----  | -------- |
| Random Flip | True |
| Random Crop | 8% ~ 100% |
| Aspect Ratio| 3/4 ~ 4/3 |
| Random PCA Lighting | 0.1 |

**Note**: We apply weight decay to all weights and biases instead of just the weights of the convolution layers.

### Training
To train the model in 8 GPUs with batch size 256, simply run
```
python main.py --config "cfgs/local_test.yaml" IMAGENET_ROOT_PATH
```

### Models
|    Model   | Top-1  | Top-5  | Baidu Yun | Google Cloud |
| ---------- | ------ | -----  | --------- | ------------ |
| FishNet99  | 77.41% | 93.59% | [Click](https://pan.baidu.com/s/11U3sRod1VfbDBRbmXph6KA)| [Click](https://www.dropbox.com/s/hvojbdsad5ue7yb/fishnet99_ckpt.tar?dl=0) | 
| FishNet150 | 78.14% | 93.95% | [Click](https://pan.baidu.com/s/1uOEFsBHIdqpDLrbfCZJGUg)| [Click](https://www.dropbox.com/s/hjadcef18ln3o2v/fishnet150_ckpt.tar?dl=0)
| FishNet201 | 78.76% | 94.39% | Available Soon | Available Soon |

### TODO:
- [ ] Update our arxiv paper.
- [x] Release pre-train models.
- [ ] Train the model with more training tricks.

### Citation

If you find our research useful, please cite the paper:
```
@inproceedings{sun2018fishnet,
  title={FishNet: A Versatile Backbone for Image, Region, and Pixel Level Prediction},
  author={Sun, Shuyang and Pang, Jiangmiao and Shi, Jianping and Yi, Shuai and Ouyang, Wanli},
  booktitle={Advances in Neural Information Processing Systems},
  pages={760--770},
  year={2018}
}
```

### Contact
You can contact Shuyang Sun by sending email to shuyang.sun@sydney.edu.au.
We will release the code and details ASAP.