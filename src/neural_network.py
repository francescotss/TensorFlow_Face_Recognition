import tensorflow as tf
import numpy as np
import PIL
import PIL.Image

import matplotlib.pyplot as plt

from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

config = {}


def read_config():
    file = open("../config/config.cfg", "r")
    for line in file:
        if line[0] == '#' or line[0] == '\n':
            continue
        key, value = line.split("=")
        config[key] = value.rstrip()

    print(config)


def load_data(category):
    def process_line(line):
        line_parts = tf.strings.split(line, ";")
        image_path = line_parts[0]
        label = line_parts[1]
        label = tf.strings.to_number(label, out_type=tf.int32)
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image)
        return image, label

    dataset = tf.data.TextLineDataset(config[category])
    dataset = dataset.map(process_line, num_parallel_calls=AUTOTUNE)

    return dataset


def preprocess_data(dataset, augment=False):
    rescale = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    ])

    augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    dataset = dataset.map(lambda x, y: (rescale(x), y), num_parallel_calls=AUTOTUNE)
    if augment:
        dataset = dataset.map(lambda x, y: (augmentation(x), y), num_parallel_calls=AUTOTUNE)

    # Performance improvements
    dataset.cache()
    dataset.shuffle(buffer_size=1000)
    dataset.batch(config["BATCH_SIZE"])
    dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset


def main():
    read_config()
    train_ds = load_data("TRAINING_CSV")
    train_ds = preprocess_data(train_ds, True)
    return


main()
