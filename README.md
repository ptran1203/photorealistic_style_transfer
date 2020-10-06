# photorealistic_style_transfer

Photorealistic Style Transfer via Wavelet Transforms - https://arxiv.org/abs/1903.09760

Keras + tensorflow implementation of WCT2.

Original implementation in [PyTorch](https://github.com/clovaai/WCT2) by [Jaejun-yoo](https://github.com/jaejun-yoo)

------

Dataset(.pkl) and weights can be found [here](https://drive.google.com/drive/folders/16iG3RGRUbRBn0h9fydwlAZtjFzSV1eUg?usp=sharing) (You can add path to your google drive if you want to run on [colab](/WCT2.ipynb))

------

#### results

| Content | Style | Result |
|--|--|--|
|![c1](/examples/input/in29.png)|![g1](/examples/style/tar29.png)| ![g1](/examples/output/out29.png) |
|![c1](/examples/input/in39.png)|![g1](/examples/style/tar39.png)| ![g1](/examples/output/out39.png) |
|![c1](/examples/input/in41.png)|![g1](/examples/style/tar41.png)| ![g1](/examples/output/out41.png) |
|![c1](/examples/input/in43.png)|![g1](/examples/style/tar43.png)| ![g1](/examples/output/out43.png) |
|![c1](/examples/input/in31.png)|![g1](/examples/style/tar31.png)| ![g1](/examples/output/out31.png) |
|![c1](/examples/input/in17.png)|![g1](/examples/style/tar17.png)| ![g1](/examples/output/out17.png) |
|![c1](/examples/input/in60.png)|![g1](/examples/style/tar60.png)| ![g1](/examples/output/out60.png) |
|![c1](/examples/input/in42.png)|![g1](/examples/style/tar42.png)| ![g1](/examples/output/out42.png) |

#### Without segmentation map, model can't capture the object well
|![c1](/examples/input/in20.png)|![g1](/examples/style/tar20.png)| ![g1](/examples/output/out20.png) |
|--|--|--|
