import tensorflow as tf
import tensorflow.keras.layers as KL


class ConstLayer(KL.Layer):
    def __init__(self, x, name=None, **kwargs):
        super(ConstLayer, self).__init__(name=name, **kwargs)
        self.x = tf.Variable(x)

    def call(self, input):
        return self.x

    def get_config(self):
        config = super(ConstLayer, self).get_config()
        return config

