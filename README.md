# photorealistic_style_transfer  <a href="https://colab.research.google.com/github/ptran1203/pytorch-animeGAN/blob/master/notebooks/animeGAN_inference.ipynb" target="_parent"><img src="https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667" alt="Open In Colab" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg"></a>

Photorealistic Style Transfer via Wavelet Transforms - https://arxiv.org/abs/1903.09760

Keras + tensorflow implementation of WCT2.

Original implementation in [PyTorch](https://github.com/clovaai/WCT2) by [Jaejun-yoo](https://github.com/jaejun-yoo)

## 1. Usage

### 1.1 Download dataset

```
wget -O https://github.com/ptran1203/photorealistic_style_transfer/releases/download/v1.0/tfrecords.zip
unzip tfrecords.zip
```

### 1.2 Train

```
python3 train.py --train-tfrec tfrecords/train.tfrec\
                 --epochs 10
                 --batch-size 8
                 --checkpoint-path /content/ckp/wtc2.h5     # Save to this path
                 --resume                                   # Continue training
                 --lr 1e-4
```
### 1.3 Inference

```
!python3 inference.py --content /content/photorealistic_style_transfer/examples/input/in21.png\
                    --style /content/photorealistic_style_transfer/examples/style\
                    --output /content/tests
```


## 2. Results

| Content | Style | Result |
|--|--|--|
|![c1](/examples/input/in17.png)|![g1](/examples/style/tar17.png)| ![g1](/examples/output/out17.png) |
|![c1](/examples/input/in29.png)|![g1](/examples/style/tar29.png)| ![g1](/examples/output/out29.png) |
|![c1](/examples/input/in31.png)|![g1](/examples/style/tar31.png)| ![g1](/examples/output/out31.png) |
|![c1](/examples/input/in35.png)|![g1](/examples/style/tar35.png)| ![g1](/examples/output/out35.png) |
|![c1](/examples/input/in39.png)|![g1](/examples/style/tar39.png)| ![g1](/examples/output/out39.png) |
|![c1](/examples/input/in43.png)|![g1](/examples/style/tar43.png)| ![g1](/examples/output/out43.png) |
|![c1](/examples/input/in46.png)|![g1](/examples/style/tar46.png)| ![g1](/examples/output/out46.png) |
|![c1](/examples/input/in52.png)|![g1](/examples/style/tar52.png)| ![g1](/examples/output/out52.png) |
|![c1](/examples/input/in55.png)|![g1](/examples/style/tar55.png)| ![g1](/examples/output/out55.png) |

#### Without segmentation map, model failed to transfer the images properly
|![c1](/examples/input/in20.png)|![g1](/examples/style/tar20.png)| ![g1](/examples/output/out20.png) |
|--|--|--|
