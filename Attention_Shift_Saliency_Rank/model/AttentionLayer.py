from keras.layers import *
import tensorflow as tf
from fpn_network import utils


class AttentionLayer(Layer):
    def __init__(self, config, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        theta_feat = inputs[0]
        phi_feat = inputs[1]
        g_feat = inputs[2]

        names = ["attn_feat"]
        outputs = utils.batch_slice(
            [theta_feat, phi_feat, g_feat],
            lambda w, x, y: attention_graph(w, x, y, self.config),
            self.config.BATCH_NUM, names=names)

        outputs = tf.reshape(outputs, (-1, self.config.SAL_OBJ_NUM, self.config.BOTTLE_NECK_SIZE // self.config.NUM_ATTN_HEADS))

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.config.SAL_OBJ_NUM, self.config.BOTTLE_NECK_SIZE // self.config.NUM_ATTN_HEADS)


def attention_graph(theta_feat, phi_feat, g_feat, config):
    theta_feat = tf.expand_dims(theta_feat, axis=0)
    phi_feat = tf.expand_dims(phi_feat, axis=0)
    g_feat = tf.expand_dims(g_feat, axis=0)

    attn = dot([theta_feat, phi_feat], axes=2)

    sqrt_dim = np.sqrt(config.BOTTLE_NECK_SIZE // config.NUM_ATTN_HEADS)

    attn = (1. / sqrt_dim) * attn

    attn = Lambda(lambda z: activations.softmax(z, axis=1))(attn)

    attn = dot([attn, g_feat], axes=[2, 1])

    attn = tf.squeeze(attn, axis=0)

    return attn
