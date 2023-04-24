import tensorflow as tf
from tensorflow import keras
from keras import models, layers, optimizers, utils


def gen_samples(ds_train, n, psi=16, t=256):
    return [
        list(ds_train.shuffle(n).take(psi).batch(psi).as_numpy_iterator())[0][0]
        for _ in range(t)
    ]


def _tf_ann(X, samples, soft_rate=1000, p=2, soft=True):
    m_dis = None
    for i in range(samples.shape[0]):
        i_sample = samples[i : i + 1, :]
        l_dis = tf.math.reduce_sum((X - i_sample) ** p, axis=1, keepdims=True) ** (
            1 / p
        )
        if m_dis is None:
            m_dis = l_dis
        else:
            m_dis = tf.concat([m_dis, l_dis], 1)

    m_dis = soft_rate * m_dis

    if soft:
        feature_map = tf.nn.softmax(-m_dis, axis=0)
    else:
        feature_map = tf.one_hot(tf.math.argmax(-m_dis, axis=1), samples.shape[0])
    # l_dis_min = tf.math.reduce_sum(m_dis * feature_map, axis=0)
    return feature_map


class IsolationEncodingLayer(layers.Layer):
    def __init__(self, samples, p=2, soft=True, **kwargs):
        super(IsolationEncodingLayer, self).__init__(**kwargs)
        self.samples = samples
        self.p = p
        self.soft = soft

    def call(self, inputs):
        return _tf_ann(inputs, self.samples, p=self.p, soft=self.soft)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "samples": self.samples,
                "p": self.p,
                "soft": self.soft,
            }
        )
        return config


def _build_ann_model(
    dims,
    n_classes,
    t_samples,
    p=2,
    soft=True,
    optimizer=optimizers.Adam(learning_rate=1e-3),
):
    t = len(t_samples)
    if t <= 0:
        raise ValueError("t <= 0")
    _, dims = t_samples[0].shape

    inputs = keras.Input(name="inputs_x", shape=(dims,))
    lambdas = [
        IsolationEncodingLayer(t_samples[i], p=p, soft=soft, name="ann_{}".format(i))(
            inputs
        )
        for i in range(t)
    ]
    concatenated = layers.Concatenate(axis=1, name="concatenated")(lambdas)
    outputs = layers.Dense(units=n_classes, activation="softmax", name="outputs_y")(
        concatenated
    )

    model = keras.Model(name="isolation_encoding", inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=optimizer,
        loss="hinge",
        metrics=["categorical_accuracy"],
    )
    return model


def train_ann(
    train_data,
    validation_data,
    t_samples,
    dims,
    n_classes,
    save_dir,
    p=2,
    epochs=30,
    batch_size=64,
    soft=True,
    optimizer=optimizers.Adam(learning_rate=1e-3),
    callbacks=[],
):
    model = _build_ann_model(
        dims, n_classes, t_samples, p=p, soft=soft, optimizer=optimizer
    )

    model.fit(
        train_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=[
            keras.callbacks.TensorBoard(log_dir=save_dir + "/log", histogram_freq=1)
        ]
        + callbacks,
    )
    model.save(save_dir + "/model")


import numpy as np
from sklearn import base


class ANNClassifier(base.BaseEstimator, base.ClassifierMixin):
    def __init__(
        self,
        t_samples,
        p=2,
        soft=True,
        epochs=100,
        batch_size=None,
        keras_optimizer=optimizers.Adam(learning_rate=1e-3),
        keras_callbacks=[],
        verbose=0,
    ):
        self.t_samples = t_samples
        self.p = p
        self.soft = soft
        self.epochs = epochs
        self.batch_size = batch_size
        self.keras_model_ = None
        self.keras_optimizer = keras_optimizer
        self.keras_callbacks = keras_callbacks
        self.verbose = verbose

    def fit(self, X, y):
        if self.keras_model_ is None:
            self.keras_model_ = _build_ann_model(
                X.shape[1],
                len(np.unique(y)),
                self.t_samples,
                self.p,
                self.soft,
                self.keras_optimizer,
            )
        if self.batch_size is None:
            self.batch_size = X.shape[0]
        self.history_ = self.keras_model_.fit(
            X,
            utils.to_categorical(y),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=self.keras_callbacks,
            verbose=self.verbose,
        )
        return self

    def predict(self, X):
        results = self.keras_model_(X)
        results = results.numpy().argmax(axis=1)
        return results
