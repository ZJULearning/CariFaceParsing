# CariFaceParsing
Code for ICIP2019 paper：Weakly-supervised Caricature Face Parsing through Domain Adaptation

## Authors
Wenqing Chu, Wei-Chih Hung, Yi-Hsuan Tsai, Deng Cai, Ming-Hsuan Yang

## Introduction
A caricature is an artistic form of a person's picture in which certain striking characteristics are abstracted or exaggerated in order to create a humor or sarcasm effect. For numerous caricature related applications such as attribute recognition and caricature editing, face parsing is an essential pre-processing step that provides a complete facial structure understanding. However, current state-of-the-art face parsing methods require large amounts of labeled data on the pixel-level and such process for caricature is tedious and labor-intensive. For real photos, there are numerous labeled datasets for face parsing. Thus, we formulate caricature face parsing as a domain adaptation problem, where real photos play the role of the source domain, adapting to the target caricatures. Specifically, we first leverage a spatial transformer based network to enable shape domain shifts. A feed-forward style transfer network is then utilized to capture texture-level domain gaps. With these two steps, we synthesize face caricatures from real photos, and thus we can use parsing ground truths of the original photos to learn the parsing model. Experimental results on the synthetic and real caricatures demonstrate the effectiveness of the proposed domain adaptation algorithm.

## Datasets
- [Helen](http://www.ifp.illinois.edu/~vuongle2/helen/): contains the facial photos as source domain.
- [Webcaricature](https://cs.nju.edu.cn/rl/WebCaricature.htm): contains the facial caricaturess as target domain.

# Dependency
- PyTorch = 0.4.1
- Python = 3.6

## Usage
Our method has two parts, adaptation and segmentation.
For adaptation, go to the adaptation directory.
Please put the Webcaricature dataset to "CariFaceParsing/adaptation/datasets/face\_webcaricature". And link "trainA" and "val" to "photo", link "trainB", "trainC", "trainD", "trainE", "trainF", "trainG", "trainH", "trainI" to "caricature".
And download the provided "landmark\_webcaricature" and put it to "CariFaceParsing/adaptation/datasets/"

```
python train.py --dataroot ./datasets/landmark_webcaricature/ --name shape_adaptation --model shape_adaptation --input_nc 20 --output_nc 11 --dataset_mode star --batch_size 8 
```

Then put Helen dataset to "CariFaceParsing/adaptation/datasets/helen" and it should have three subfolder, "images", "labels", "landmark".

```
generate_shape_adaptation.py --dataroot ./datasets/helen/landmark/ --name shape_adaptation --model shape_adaptation --input_nc 20 --output_nc 11 --dataset_mode helen

```

Put the adapted results to "CariFaceParsing/adaptation/datasets/helen\_shape\_adaptation" and it should have "images", "labels". Put the provided "train\_style" and "test\_style" here and link "train\_content", "test\_content" to images.



```
python train.py --dataroot ./datasets/helen_shape_adaptation --name style_adaptation --model style_adaptation --input_nc 3 --output_nc 3 --dataset_mode style --batch_size 1
```


```
python generate_style_adaptation.py --dataroot ./datasets/helen_shape_adaptation/ --name style_adaptation --model style_adaptation --input_nc 3 --output_nc 3 --dataset_mode style
```

For segmentation, go to the Segmentation directory.
Please install by
```
python setup.py install --prefix=~/anaconda3/
```



You can use the generated images and labels as training data. In our experiment, we randomly select around 1000 samples as validation data to find the best parsing model. Please put the train and val data to CariFaceParsing/Segmentation/dataset/helen/. And put the test data (Webcaricature) to CariFaceParsing/Segmentation/dataset/cari/. 

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset helen --model psp --backbone resnet50 --aux --batch-size 16 --output_nc --epoch 30
```
We provide the mannually annotated caricature dataset in Google Drive (https://drive.google.com/open?id=1x0bJ7wBAsC_jSjm30SIqjl-huuBZMKh6)

```
python test.py --dataset cari --model psp --backbone resnet50 --aux --eval --resume=runs/helen/psp/default/model_best.pth.tar
```

We also provide the pretrained segmentation, style adaptation and shape adaptation models in Google Drive (https://drive.google.com/open?id=1x0bJ7wBAsC_jSjm30SIqjl-huuBZMKh6)


## Reference
If you use the code or our dataset, please cite our paper

@article{Chu2019Weakly,
    title={Weakly-supervised Caricature Face Parsing through Domain Adaptation},
    author={Chu, Wenqing and Hung, Wei-Chih and Tsai, Yi-Hsuan and Cai, Deng and Yang, Ming-Hsuan},
    journal={arXiv preprint arXiv:1905.05091},
    year={2019},
}

## Acknowledgment
This code is heavily borrowed from
- [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
- [CycleGan](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)




