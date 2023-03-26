import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds

epochs = 30
batch_size = 64
image_w, image_h = 32, 32

(ds_train_raw, ds_test_raw), ds_info = tfds.load(
    "malaria",
    split=["train[:80%]", "train[80%:]"],
    shuffle_files=False,
    as_supervised=True,
    with_info=True,
)

n_classes = ds_info.features["label"].num_classes
n = ds_info.splits["train"].num_examples


def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = layers.Resizing(image_h, image_w)(image)
    image = tf.reshape(image, [-1])
    label = tf.one_hot(tf.cast(label, tf.int32), n_classes)
    label = tf.cast(label, tf.float32)
    return image, label


ds_train_normalized = ds_train_raw.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE
).cache()

ds_test_normalized = ds_test_raw.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE
).cache()


def prepare(ds, batch_size=batch_size):
    return ds.shuffle(n).batch(batch_size).prefetch(tf.data.AUTOTUNE)


dims = list(ds_train_normalized.take(1))[0][0].shape[0]

print("n: ", n, "n_classes: ", n_classes, "dims: ", dims)


# def minmax_reducer(current, input):
#     X, _ = input
#     return (
#         tf.reduce_min([current[0], X], axis=0),
#         tf.reduce_max([current[0], X], axis=0),
#     )


# x0, _ = list(ds_train_normalized.take(1))[0]
# min_train, max_train = ds_train_normalized.reduce((x0, x0), minmax_reducer)
