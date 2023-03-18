from tensorflow import keras
from keras import layers

RandomFourierFeatures = keras.layers.experimental.RandomFourierFeatures


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
