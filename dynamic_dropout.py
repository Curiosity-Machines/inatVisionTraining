import tensorflow as tf
from tensorflow.keras import layers

class DynamicDropout(layers.Layer):
    def __init__(self, initial_rate=0.5, **kwargs):
        super(DynamicDropout, self).__init__(**kwargs)
        self.rate = tf.Variable(initial_rate, trainable=False, dtype=tf.float32)

    def call(self, inputs, training=None):
        if training:
            if inputs.dtype != tf.float32:
                inputs = tf.cast(inputs, tf.float32)
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs

    def set_rate(self, new_rate):
        self.rate.assign(new_rate)
