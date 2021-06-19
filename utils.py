import numpy as np
import urllib.request
import cv2
import matplotlib.pyplot as plt
import os
import urllib.request
from tqdm import tqdm

HTTP_PREFIXES = [
    'http',
    'data:image/jpeg',
]

ASSET_HOST = 'https://github.com/ptran1203/photorealistic_style_transfer/releases/download/v1.0'

def http_get_img(url, rst=64):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if rst:
        img = image_resize(img, rst)

    return img


def get_local_img(path, rst=None):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if rst:
        img = image_resize(img, rst)

    return img


def read_img(path, rst, expand_dims=False):
    if any(path.startswith(prefix) for prefix in HTTP_PREFIXES):
        img = http_get_img(path, rst)
    else:
        img = get_local_img(path, rst)

    if expand_dims:
        img = np.expand_dims(expand_dims, 0)

    return img


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def display_outputs(content, style, output=None, figsize=(15, 8)):
    fig = plt.figure(figsize=figsize)
    plt.axis("off")
    fig.add_subplot(1, 2, 1)
    plt.imshow(content / 255.0)

    fig.add_subplot(1, 2, 2)
    plt.imshow(style / 255.0)
    plt.show()

    if output is not None:
        plt.imshow(output / 255.0)
        plt.show()


class DownloadProgressBar(tqdm):
    '''
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
    '''
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_weight():
    '''
    Download weight and save to local file
    '''
    filename = 'wtc2.h5'
    os.makedirs('.cache', exist_ok=True)

    url = f'{ASSET_HOST}/{filename}'
    save_path = f'.cache/{filename}'

    if os.path.isfile(save_path):
        return save_path

    desc = f'Downloading {url} to {save_path}'
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, save_path, reporthook=t.update_to)

    return save_path
