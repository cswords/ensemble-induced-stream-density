import tensorflow as tf
from tensorflow import keras
from keras import layers


def _tf_ann_weighted(X, samples, sample_weights, p=2, soft=True):
    m_dis = None  # [n, psi]
    for i in range(samples.shape[0]):
        i_sample = samples[i : i + 1, :]  # [i, dims]
        l_dis = tf.math.reduce_sum((X - i_sample) ** p, axis=1, keepdims=True) ** (
            1 / p
        )  # [n, 1]
        if m_dis is None:
            m_dis = l_dis
        else:
            m_dis = tf.concat([m_dis, l_dis], 1)

    m_dis = m_dis * sample_weights

    if soft:
        feature_map = tf.nn.softmax(-m_dis, axis=0)
    else:
        feature_map = tf.one_hot(tf.math.argmax(-m_dis, axis=1), samples.shape[0])
    # l_dis_min = tf.math.reduce_sum(m_dis * feature_map, axis=0)
    return feature_map


class FlexibleIsolationEncodingLayer(layers.Layer):
    def __init__(self, samples, p=2, **kwargs):
        super(FlexibleIsolationEncodingLayer, self).__init__(**kwargs)
        self.samples = samples
        self.p = p

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.sample_weights = self.add_weight(
            name="dimential_weights",
            shape=(
                1,
                self.samples.shape[0],
            ),
            initializer="ones",
            trainable=True,
        )
        super(FlexibleIsolationEncodingLayer, self).build(input_shape)

    def call(self, inputs):
        return _tf_ann_weighted(inputs, self.samples, self.sample_weights, self.p)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "samples": self.samples,
                "p": self.p,
            }
        )
        return config


def train_ann_weighted(
    train_data,
    validation_data,
    t_samples,
    dims,
    n_classes,
    save_dir,
    p=2,
    epochs=30,
    batch_size=64,
):
    t = len(t_samples)
    if t <= 0:
        raise ValueError("t <= 0")
    _, dims = t_samples[0].shape

    inputs = keras.Input(name="inputs_x", shape=(dims,))
    lambdas = [
        FlexibleIsolationEncodingLayer(t_samples[i], p=p, name="ann_flex_{}".format(i))(
            inputs
        )
        for i in range(t)
    ]
    concatenated = layers.Concatenate(axis=1, name="concatenated")(lambdas)
    outputs = layers.Dense(units=n_classes, name="outputs_y")(concatenated)

    model = keras.Model(name="isolation_encoding", inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.hinge,
        metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
    )

    model.fit(
        train_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=[
            keras.callbacks.TensorBoard(log_dir=save_dir + "/log", histogram_freq=1)
        ],
    )
    model.save(save_dir + "/model")
