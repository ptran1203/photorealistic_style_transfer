import argparse
import os
import tensorflow as tf
import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str)
    parser.add_argument('--output-dir', type=str)

    return parser.parse_args()

def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def image_example(id_, image_string, label):
    feature = {
        "image_id": bytes_feature(id_),
        "label": bytes_feature(label),
        "image": bytes_feature(image_string),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def write_tfrecords(data, file_path):
    with tf.io.TFRecordWriter(file_path) as writer:
        for i, item in tqdm(data.iterrows(), total=len(data)):
            image_string = open(item.path, "rb").read()
            tf_example = image_example(
                item["image"].encode("utf-8"),
                image_string,
                item["label"].encode("utf-8"),
            )
            writer.write(tf_example.SerializeToString())


def create_df(input_dir, folder):
    img_list = []
    label_list = []
    img_paths = []
    train_dir = os.path.join(input_dir, folder)
    for label in os.listdir(train_dir):
        img_dir = os.path.join(train_dir, label)
        imgs = os.listdir(img_dir)
        print(label, len(imgs))
        for img in imgs:
            img_paths.append(os.path.join(img_dir, img))
            img_list.append(img)
            label_list.append(label)
            
    return pd.DataFrame({
        "image": img_list,
        "path": img_paths,
        "label": label_list,
    })

train_df = create_df("images/images")
val_df = create_df("validation/validation")

train_df = train_df.iloc[:10000]
f"Train images {len(train_df)}, validation images {len(val_df)}"

def main(args):
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(args.input_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    




if __name__ == '__main__':
    main(parse_args())