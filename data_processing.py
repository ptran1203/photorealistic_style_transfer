import numpy as np
import utils
import tensorflow as tf
from tensorflow.python.keras.applications import imagenet_utils

AUTOTUNE = tf.data.experimental.AUTOTUNE

image_feature_description = {
    "image_id": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
}

def preprocess_image(x):
    # https://www.tensorflow.org/api_docs/python/tf/keras/applications/vgg16/preprocess_input
    x = imagenet_utils.preprocess_input(x, mode="caffe")

    # Scaling to -1 1
    # x -= 127.5
    # x /= 127.5

    return x

def deprocess_image(x):
    """
    Custom implementation
    https://github.com/tensorflow/tensorflow/blob/85c8b2a817f95a3e979ecd1ed95bff1dc1335cff/tensorflow/python/keras/applications/imagenet_utils.py#L242
    """
    x = x[..., ::-1]
    return x


def preprocess_input(x, y):
    return preprocess_image(x), preprocess_image(y)

def decode_sample(example):
    sample = tf.io.parse_single_example(example, image_feature_description)
    image = tf.image.decode_jpeg(sample["image"], channels=3)
    label = sample["label"]
    image_id = sample["image_id"]

    image = tf.image.resize(image, (256, 256))

    # This model is trained to recontruct the original image
    # If you want label, replace with
    # return image, label

    return image, image

def build_input_pipe(tfrecord_file, batch_size=0, preprocess_method="vgg19", repeat=False):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(decode_sample)

    if batch_size:
        dataset = dataset.batch(batch_size)

    dataset = dataset.map(
        preprocess_input,
        num_parallel_calls=AUTOTUNE
    )

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


def restore_image(x, data_format='channels_last'):
    mean = [103.939, 116.779, 123.68]

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] += mean[0]
            x[1, :, :] += mean[1]
            x[2, :, :] += mean[2]
        else:
            x[:, 0, :, :] += mean[0]
            x[:, 1, :, :] += mean[1]
            x[:, 2, :, :] += mean[2]
    else:
        x[..., 0] += mean[0]
        x[..., 1] += mean[1]
        x[..., 2] += mean[2]

    if data_format == 'channels_first':
        # 'BGR'->'RGB'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'BGR'->'RGB'
        x = x[..., ::-1]

    return rescale(x) * 255.0


def rescale(x):
    """
    Rescale input to range [0, 1]
    """
    min_val = np.min(x)
    max_val = np.max(x)
    x = (x - min_val) / (max_val - min_val)

    return x
