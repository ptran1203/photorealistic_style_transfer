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
    return x


def preprocess_input(x, y):
    return preprocess_image(x), preprocess_image(y)


def decode_sample(example):
    sample = tf.io.parse_single_example(example, image_feature_description)
    image = tf.image.decode_jpeg(sample["image"], channels=3)
    image = tf.image.resize(image, (256, 256))

    return image, image


def build_input_pipe(tfrecord_file, batch_size=0, repeat=False):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(decode_sample)

    if batch_size:
        dataset = dataset.batch(batch_size)

    dataset = dataset.map(
        preprocess_input,
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.apply(tf.data.experimental.ignore_errors())

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.prefetch(AUTOTUNE)

    return dataset

