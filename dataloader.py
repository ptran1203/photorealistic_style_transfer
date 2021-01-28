import numpy as np
import utils
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
image_feature_description = {
    "image_id": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
}

def preprocess_input(x, y):
    return (
        tf.keras.applications.vgg19.preprocess_input(x),
        tf.keras.applications.vgg19.preprocess_input(y),
    )

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

def build_input_pipe(tfrecord_file, batch_size=0, preprocess_method="vgg19"):
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    dataset = dataset.map(decode_sample)

    if batch_size:
        dataset = dataset.batch(batch_size)

    dataset = dataset.map(
        preprocess_input,
        num_parallel_calls=AUTOTUNE
    )

    return dataset
