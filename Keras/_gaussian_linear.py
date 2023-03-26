from tensorflow import keras
from keras import layers

RandomFourierFeatures = keras.layers.experimental.RandomFourierFeatures


def _build_gaussian_linear_model(
    dims, n_classes, output_features=2000, output_scale=10.0
):
    model_svm = keras.Sequential(
        [
            layers.Input(shape=(dims,)),
            RandomFourierFeatures(
                output_dim=output_features,
                scale=output_scale,
                kernel_initializer="gaussian",
            ),
            layers.Dense(units=n_classes),
        ]
    )
    model_svm.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.hinge,
        metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
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
):
    model_svm = _build_gaussian_linear_model(
        dims, n_classes, output_features, output_scale
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
        ],
    )
    model_svm.save(save_dir + "/model")


import numpy as np
from sklearn import base


class GaussianLinearClassifier(base.BaseEstimator, base.ClassifierMixin):
    def __init__(self, output_features=2000, output_scale=10.0):
        self.output_features = output_features
        self.output_scale = output_scale
        self.keras_model = None

    def fit(self, X, y):
        if self.keras_model is None:
            self.keras_model = _build_gaussian_linear_model(
                X.shape[1], len(np.unique(y)), self.output_features, self.output_scale
            )
        self.keras_model.fit(
            X,
            y,
            epochs=30,
            batch_size=64,
        )
        return self

    def predict(self, X):
        results = self.keras_model(X)
        return results
