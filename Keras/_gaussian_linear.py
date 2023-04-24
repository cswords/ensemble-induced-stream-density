from tensorflow import keras
from keras import models, layers, optimizers, utils

RandomFourierFeatures = keras.layers.experimental.RandomFourierFeatures

def _build_gaussian_linear_model(
    dims,
    n_classes,
    output_features=4096,
    output_scale=10.0,
    optimizer=optimizers.Adam(learning_rate=1e-3),
):
    model_svm = models.Sequential(
        [
            layers.Input(shape=(dims,)),
            RandomFourierFeatures(
                output_dim=output_features,
                scale=output_scale,
                kernel_initializer="gaussian",
                # trainable=True,
            ),
            layers.Dense(units=n_classes, activation="softmax"),
        ]
    )
    model_svm.compile(
        optimizer=optimizer,
        loss="hinge",
        metrics=["categorical_accuracy"],
    )
    return model_svm


def train_gaussian_linear(
    train_data,
    validation_data,
    dims,
    n_classes,
    save_dir,
    output_features=2000,
    output_scale=10.0,
    epochs=30,
    batch_size=64,
    optimizer=optimizers.Adam(learning_rate=1e-3),
    callbacks=[],
):
    model_svm = _build_gaussian_linear_model(
        dims,
        n_classes,
        output_features,
        output_scale,
        optimizer,
    )

    model_svm.fit(
        train_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=[
            keras.callbacks.TensorBoard(
                log_dir=save_dir + "/log",
                histogram_freq=1,
            )
        ]
        + callbacks,
    )
    model_svm.save(save_dir + "/model")


import numpy as np
from sklearn import base


class GaussianLinearClassifier(base.BaseEstimator, base.ClassifierMixin):
    def __init__(
        self,
        output_features=2000,
        output_scale=10.0,
        epochs=100,
        batch_size=None,
        keras_optimizer=optimizers.Adam(learning_rate=1e-3),
        keras_callbacks=[],
        verbose=0,
    ):
        self.output_features = output_features
        self.output_scale = output_scale
        self.epochs = epochs
        self.batch_size = batch_size
        self.keras_model_ = None
        self.keras_optimizer = keras_optimizer
        self.keras_callbacks = keras_callbacks
        self.verbose = verbose

    def fit(self, X, y):
        if self.keras_model_ is None:
            self.keras_model_ = _build_gaussian_linear_model(
                X.shape[1],
                len(np.unique(y)),
                self.output_features,
                self.output_scale,
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
        self.results_ = self.keras_model_.predict(X, verbose=self.verbose)
        return self.results_.argmax(axis=-1)
