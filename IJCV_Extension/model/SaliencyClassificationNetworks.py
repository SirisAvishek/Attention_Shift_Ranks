from tensorflow.keras.layers import *
from model.BatchNorm import BatchNorm


def sal_class_network(input_feature, config, dropout=0.5):
    dim1, dim2, dim3 = config.RANK_CLASS_HIDDEN_LAYER_DIMS

    x = TimeDistributed(Dense(dim1), name="sal_class_mod_dense_1")(input_feature)
    x = TimeDistributed(BatchNorm(), name="sal_class_mod_bn_1")(x, training=config.TRAIN_BN)
    x = Activation("relu")(x)
    x = TimeDistributed(Dropout(dropout), name="sal_class_mod_dropout_1")(x)

    x = TimeDistributed(Dense(dim2), name="sal_class_mod_dense_2")(x)
    x = TimeDistributed(BatchNorm(), name="sal_class_mod_bn_2")(x, training=config.TRAIN_BN)
    x = Activation("relu")(x)
    x = TimeDistributed(Dropout(dropout), name="sal_class_mod_dropout_2")(x)

    x = TimeDistributed(Dense(dim3), name="sal_class_mod_dense_3")(x)
    x = TimeDistributed(BatchNorm(), name="sal_class_mod_bn_3")(x, training=config.TRAIN_BN)
    x = Activation("relu")(x)
    x = TimeDistributed(Dropout(dropout), name="sal_class_mod_dropout_3")(x)

    # ------------

    # Saliency Ranking
    num_classes = 1
    sal_rank_x = TimeDistributed(Dense(num_classes), name="sal_rank")(x)

    return sal_rank_x

