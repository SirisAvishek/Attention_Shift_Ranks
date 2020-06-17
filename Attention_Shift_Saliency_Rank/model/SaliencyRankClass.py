from keras.models import *
from keras.layers import *
from fpn_network.BatchNorm import BatchNorm


def build_rank_class_model(config):
    model_input = Input(shape=(config.RANK_FEAT_SIZE,), name="input_rank_class")

    dim1, dim2, dim3 = config.RANK_CLASS_HIDDEN_LAYER_DIMS

    x = Dense(dim1, name="rank_class_mod_dense_1")(model_input)
    x = BatchNorm(name="rank_class_mod_bn_1")(x, training=config.TRAIN_BN)
    x = Activation("relu")(x)

    x = Dense(dim2, name="rank_class_mod_dense_2")(x)
    x = BatchNorm(name="rank_class_mod_bn_2")(x, training=config.TRAIN_BN)
    x = Activation("relu")(x)

    x = Dense(dim3, name="rank_class_mod_dense_3")(x)
    x = BatchNorm(name="rank_class_mod_bn_3")(x, training=config.TRAIN_BN)
    x = Activation("relu")(x)

    x = Dense(config.SAL_RANK, activation="softmax", name="rank_class")(x)

    score_model = Model(inputs=[model_input], outputs=x, name="point_wise_scoring_model")

    return score_model
