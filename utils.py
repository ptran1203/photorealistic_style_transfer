import numpy as np
import urllib.request
import cv2
import matplotlib.pyplot as plt

def http_get_img(url, rst=64):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if rst:
        img = image_resize(img, rst)

    img = np.expand_dims(img, 0)
    return img


def get_local_img(path, rst=None):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if rst:
        img = image_resize(img, rst)

    img = np.expand_dims(img, 0)
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
