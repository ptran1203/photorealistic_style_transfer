import pickle
import numpy as np
import urllib.request
import cv2

MEAN_PIXCELS = np.array([103.939, 116.779, 123.68])

def pickle_save(object, path, log=False):
    try:
        log and print('save data to {} successfully'.format(path))
        with open(path, "wb") as f:
            return pickle.dump(object, f)
    except Exception as e:
        log and print('save data to {} failed'.format(path))


def pickle_load(path, log=False):
    try:
        log and print("Loading data from {} - ".format(path))
        with open(path, "rb") as f:
            data = pickle.load(f)
            log and print("DONE")
            return data
    except Exception as e:
        print(str(e))
        return None


def http_get_img(url, rst=64):
    req = urllib.request.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)

    if rst:
        img = image_resize(img, rst)

    img = np.expand_dims(img, 0)
    return img


def get_local_img(path, rst=None):
    img = cv2.imread(path)
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
