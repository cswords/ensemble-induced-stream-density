import tensorflow as tf
from tensorflow import keras
from keras import layers


def gen_samples(ds_train, n, psi, t=1000):
    return [
        list(ds_train.shuffle(n).take(psi).batch(psi).as_numpy_iterator())[0][0]
        for _ in range(t)
    ]


def _tf_ann(X, samples, p=2, soft=True):
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
        return _tf_ann(inputs, self.samples, self.p, self.soft)

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
