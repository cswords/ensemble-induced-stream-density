import os
import numpy as np
from datetime import datetime
from Data._clusters import load_mat
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from keras import layers

from Keras import (
    gen_samples,
    train_gaussian_linear,
    train_ann,
    train_ann_weighted,
)

from Common import (
    get_array_module,
    ball_scale,
    save_csv,
    save_parquet,
    init_xp,
)


data_dir = "Data/mat/clusters"
datasets = [
    # "control",
    # "dermatology",
    # "diabetesC",
    "ecoli",
    # "glass",
    "hard",
    # "hill",
    # "ILPD",
    # "Ionosphere",
    "iris",
    # "isolet",
    # "libras",
    # "liver",
    "LSVT",
    # "musk",
    "Parkinsons",
    # # "Pendig",
    # "pima",
    # "s1",
    # "s2",
    # "seeds",
    # "Segment",
    # "shape",
    # "Sonar",
    # # "spam",
    # "SPECTF",
    # "thyroid",
    # "user",
    "vowel",
    # "WDBC",
    # "wine",
]


def prepare(ds, batch_size=64):
    return ds.shuffle(n).batch(batch_size).prefetch(tf.data.AUTOTUNE)


train_split_rate = 0.8
t = 1024

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True, linewidth=150)

    time_format = "%Y%m%d%H"
    save_folder = "Data/" + datetime.now().strftime(time_format) + "_keras"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for dataset_name in datasets:
        X, y = load_mat(data_dir, dataset_name, np)

        n_total, dims = X.shape
        n = int(n_total * train_split_rate)

        X = X.astype(np.float32)
        y = preprocessing.LabelEncoder().fit_transform(y)

        ds_raw = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(n_total)

        ds_train_raw = ds_raw.take(n)
        ds_test_raw = ds_raw.skip(n)

        n_classes = np.unique(y).shape[0]

        print("n: ", n, "n_classes: ", n_classes, "dims: ", dims)

        for psi in [2, 4, 8, 16, 32, 64]:
            ds_train_normalized = ds_train_raw.cache()
            ds_test_normalized = ds_test_raw.cache()

            linear_features = psi * t
            train_gaussian_linear(
                prepare(ds_train_normalized),
                prepare(ds_test_normalized),
                dims,
                n_classes,
                save_folder + "/" + dataset_name + "/linear-g-" + str(linear_features),
                output_features=linear_features,
            )

            t_samples = [X[np.random.choice(n, psi), :] for _ in range(t)]

            train_ann(
                prepare(ds_train_normalized),
                prepare(ds_test_normalized),
                t_samples,
                dims,
                n_classes,
                save_folder + "/" + dataset_name + "/ann-hard-" + str(psi),
                soft=False,
            )

            train_ann(
                prepare(ds_train_normalized),
                prepare(ds_test_normalized),
                t_samples,
                dims,
                n_classes,
                save_folder + "/" + dataset_name + "/ann-soft-" + str(psi),
                soft=True,
            )

            train_ann_weighted(
                prepare(ds_train_normalized),
                prepare(ds_test_normalized),
                t_samples,
                dims,
                n_classes,
                save_folder + "/" + dataset_name + "/ann-weighted-" + str(psi),
            )
