# photorealistic_style_transfer

Photorealistic Style Transfer via Wavelet Transforms - https://arxiv.org/abs/1903.09760

Keras + tensorflow implementation of WCT2.

Original implementation in [PyTorch](https://github.com/clovaai/WCT2) by [Jaejun-yoo](https://github.com/jaejun-yoo)

------

Quick start in [colab](/WCT2.ipynb)

------

#### results

| Content | Style | Result |
|--|--|--|
|![c1](/examples/input/in29.png)|![g1](/examples/style/tar29.png)| ![g1](/examples/output/out29.png) |
|![c1](/examples/input/in43.png)|![g1](/examples/style/tar43.png)| ![g1](/examples/output/out43.png) |
|![c1](/examples/input/in17.png)|![g1](/examples/style/tar17.png)| ![g1](/examples/output/out17.png) |
|![c1](/examples/input/in60.png)|![g1](/examples/style/tar60.png)| ![g1](/examples/output/out60.png) |

#### Without segmentation map, model failed to transfer the images properly
|![c1](/examples/input/in20.png)|![g1](/examples/style/tar20.png)| ![g1](/examples/output/out20.png) |
|--|--|--|
