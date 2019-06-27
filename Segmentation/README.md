# PyTorch-Encoding

created by [Hang Zhang](http://hangzh.com/)

## [Documentation](http://hangzh.com/PyTorch-Encoding/)

- Please visit the [**Docs**](http://hangzh.com/PyTorch-Encoding/) for detail instructions of installation and usage. 

- Please visit the [link](http://hangzh.com/PyTorch-Encoding/experiments/segmentation.html) to examples of semantic segmentation.


Install Package

1. This branch relies on PyTorch v0.4.0, please follow the instruction to install PyTorch.

2. git clone https://github.com/wenqingchu/PyTorch-Encoding

3. checkout bracn pytorch0.4.0

4. Since I use anaconda3, I install this project by "python setup.py install --prefix=~/anaconda3/"

Train

1. There are fcn.py, deeplab.py, psp.py and encnet.py model in encoding/model/.
2. If you want to train on your own dataset, you should implement a dataset file like encoding/datasets/cityscapes.py.
3. For cityscapes dataset, put gtFine, leftImg8bit, train.txt, val.txt in ~/.encoding/data/Cityscapes/data/. 
4. Please read experiment/segmentation/option.py to adjust the hyper parameter setting. I use batch_size=8, lr=0.01, 4 gpus, psp model, resnet101, 80 epoches, and the model can achieve mIoU=0.7704 on cityscapes validation dataset. 
5. Here is an example for train psp model on cityscapes.
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_cityscapes.py --dataset cityscapes --model psp --backbone resnet101 --aux --batch-size 8

Test

CUDA_VISIBLE_DEVICES=0,1,2,3 python test_cityscapes.py --dataset cityscapes --model psp --backbone resnet101 --aux --eval --resume=runs/cityscapes/psp/default/checkpoint.pth.tar

## Citations

**Context Encoding for Semantic Segmentation** [[arXiv]](https://arxiv.org/pdf/1803.08904.pdf)  
 [Hang Zhang](http://hangzh.com/), [Kristin Dana](http://eceweb1.rutgers.edu/vision/dana.html), [Jianping Shi](http://shijianping.me/), [Zhongyue Zhang](http://zhongyuezhang.com/), [Xiaogang Wang](http://www.ee.cuhk.edu.hk/~xgwang/), [Ambrish Tyagi](https://scholar.google.com/citations?user=GaSWCoUAAAAJ&hl=en), [Amit Agrawal](http://www.amitkagrawal.com/)
```
@InProceedings{Zhang_2018_CVPR,
author = {Zhang, Hang and Dana, Kristin and Shi, Jianping and Zhang, Zhongyue and Wang, Xiaogang and Tyagi, Ambrish and Agrawal, Amit},
title = {Context Encoding for Semantic Segmentation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```

**Deep TEN: Texture Encoding Network** [[arXiv]](https://arxiv.org/pdf/1612.02844.pdf)  
  [Hang Zhang](http://hangzh.com/), [Jia Xue](http://jiaxueweb.com/), [Kristin Dana](http://eceweb1.rutgers.edu/vision/dana.html)
```
@InProceedings{Zhang_2017_CVPR,
author = {Zhang, Hang and Xue, Jia and Dana, Kristin},
title = {Deep TEN: Texture Encoding Network},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {July},
year = {2017}
}
```
