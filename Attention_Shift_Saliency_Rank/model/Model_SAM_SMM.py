from keras.models import *
from keras.layers import *
from fpn_network.BatchNorm import BatchNorm
from model import SaliencyRankClass, Losses
from model.AttentionLayer import AttentionLayer


def build_saliency_rank_model(config, mode):
    # *********************** INPUTS ***********************
    input_obj_features = Input(shape=(config.SAL_OBJ_NUM, 1, 1, config.OBJ_FEAT_SIZE), name="input_obj_feat")
    input_obj_spatial_masks = Input(shape=(config.SAL_OBJ_NUM, 32, 32, 1), name="input_obj_spatial_masks")
    input_P5_feat = Input(shape=(32, 32, 256), name="input_P5_feat")

    if mode == "training":
        input_target_rank = Input(shape=(config.SAL_OBJ_NUM,), name="input_gt_ranks")

    # *********************** PROCESS Image/P5 FEATURES ***********************
    img_feat = Conv2D(config.BOTTLE_NECK_SIZE, (3, 3), name="img_feat_conv_1")(input_P5_feat)
    img_feat = BatchNorm(name="img_feat_bn_1")(img_feat, training=config.TRAIN_BN)
    img_feat = Activation('relu')(img_feat)

    img_feat = GlobalAveragePooling2D()(img_feat)

    # *********************** SELECTIVE ATTENTION MODULE ***********************
    # Reduce dimension to BOTTLNECK
    obj_feature = TimeDistributed(Conv2D(config.BOTTLE_NECK_SIZE, (1, 1)), name="obj_feat_reduce_conv1")(input_obj_features)
    obj_feature = TimeDistributed(BatchNorm(), name='obj_feat_reduce_bn1')(obj_feature, training=config.TRAIN_BN)
    obj_feature = Activation('relu')(obj_feature)

    obj_feature = Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="obj_feat_squeeze")(obj_feature)

    sa_feat = selective_attention_module(config.NUM_ATTN_HEADS, obj_feature, img_feat, config)

    # *********************** OBJECT SPATIAL MASK MODULE ***********************
    spatial_mask_feat = object_spatial_mask_module(input_obj_spatial_masks, config)

    # CONCATENATE OBJ_FEAT_MASKS + OBJ_SPATIAL_MASKS
    obj_feature = Concatenate()([sa_feat, spatial_mask_feat])

    # *********************** FINAL OBJECT FEATURE ***********************
    # FC layer for reducing the attention features
    final_obj_feat = TimeDistributed(Dense(config.RANK_FEAT_SIZE), name="obj_final_feat_dense_1")(obj_feature)
    final_obj_feat = TimeDistributed(BatchNorm(), name='obj_final_feat_bn_1')(final_obj_feat, training=config.TRAIN_BN)
    final_obj_feat = Activation('relu')(final_obj_feat)

    # *********************** OBJECT RANK ORDER NETWORK ***********************
    # Ranking Network
    point_scoring_model = SaliencyRankClass.build_rank_class_model(config)

    # Perform Ranking
    object_rank = point_scoring_model(final_obj_feat)

    if mode == "training":
        # *********************** LOSS **********************
        # Rank Loss
        rank_loss = Lambda(lambda x: Losses.sparse_categorical_cross_entropy_pos_contrib(*x), name="rank_loss")(
            [input_target_rank, object_rank])

        # *********************** FINAL ***********************
        # Model
        inputs = [input_obj_features, input_obj_spatial_masks, input_P5_feat,
                  input_target_rank]
        outputs = [object_rank, rank_loss]
        model = Model(inputs=inputs, outputs=outputs, name="attn_shift_saliency_rank_model")
    else:
        # *********************** FINAL ***********************
        # Model
        inputs = [input_obj_features, input_obj_spatial_masks,
                  input_P5_feat]
        outputs = [object_rank]
        model = Model(inputs=inputs, outputs=outputs, name="attn_shift_saliency_rank_model")

    return model


def object_spatial_mask_module(in_obj_spatial_masks, config):
    # *********************** OBJECT SPATIAL MASKS ***********************
    obj_spa_mask = TimeDistributed(Conv2D(96, (5, 5), strides=2, padding="same"), name="obj_spatial_mask_conv_1")(
        in_obj_spatial_masks)
    obj_spa_mask = TimeDistributed(BatchNorm(), name='obj_spatial_mask_bn_1')(obj_spa_mask, training=config.TRAIN_BN)
    obj_spa_mask = Activation("relu")(obj_spa_mask)

    obj_spa_mask = TimeDistributed(Conv2D(128, (5, 5), strides=2, padding="same"), name="obj_spatial_mask_conv_2")(
        obj_spa_mask)
    obj_spa_mask = TimeDistributed(BatchNorm(), name='obj_spatial_mask_bn_2')(obj_spa_mask, training=config.TRAIN_BN)
    obj_spa_mask = Activation("relu")(obj_spa_mask)

    obj_spa_mask = TimeDistributed(Conv2D(64, (8, 8)), name="obj_spatial_mask_conv_3")(obj_spa_mask)
    obj_spa_mask = TimeDistributed(BatchNorm(), name='obj_spatial_mask_bn_3')(obj_spa_mask, training=config.TRAIN_BN)
    obj_spa_mask = Activation("relu")(obj_spa_mask)

    obj_spatial_mask_feat = Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="obj_spatial_mask_squeeze")(obj_spa_mask)

    return obj_spatial_mask_feat


def selective_attention_module(num_heads, obj_feat, img_feat, config):
    head_outputs = []
    for h in range(num_heads):
        theta_dense_name = "head_" + str(h) + "_obj_theta_dense_1"
        theta_bn_name = "head_" + str(h) + "_obj_theta_bn_1"
        phi_dense_name = "head_" + str(h) + "_img_phi_dense_1"
        phi_bn_name = "head_" + str(h) + "_img_phi_bn_1"
        g_dense_name = "head_" + str(h) + "_img_g_dense_1"
        g_bn_name = "head_" + str(h) + "_img_g_bn_1"

        proj_feat_size = config.BOTTLE_NECK_SIZE // num_heads

        # Project features
        obj_theta = TimeDistributed(Dense(proj_feat_size), name=theta_dense_name)(obj_feat)
        obj_theta = TimeDistributed(BatchNorm(), name=theta_bn_name)(obj_theta, training=config.TRAIN_BN)
        img_phi = Dense(proj_feat_size, name=phi_dense_name)(img_feat)
        img_phi = BatchNorm(name=phi_bn_name)(img_phi, training=config.TRAIN_BN)
        img_g = Dense(proj_feat_size, name=g_dense_name)(img_feat)
        img_g = BatchNorm(name=g_bn_name)(img_g, training=config.TRAIN_BN)

        # Repeat Vectors
        img_phi = RepeatVector(config.SAL_OBJ_NUM)(img_phi)
        img_g = RepeatVector(config.SAL_OBJ_NUM)(img_g)

        attn_name = "attn_layer_" + str(h)
        attn = AttentionLayer(config, name=attn_name)([obj_theta, img_phi, img_g])

        head_outputs.append(attn)

    final_attn = Concatenate()(head_outputs) if num_heads > 1 else head_outputs[0]

    # Linear
    final_attn = TimeDistributed(Dense(config.BOTTLE_NECK_SIZE), name='obj_attn_feat_dense_1')(final_attn)
    final_attn = TimeDistributed(BatchNorm(), name='obj_attn_feat_bn_1')(final_attn, training=config.TRAIN_BN)
    final_attn = Activation('relu')(final_attn)

    # Add Residual
    final_attn = Add()([final_attn, obj_feat])
    final_attn = Activation('relu')(final_attn)

    # Feed_forward
    final_obj_feat = TimeDistributed(Dense(config.BOTTLE_NECK_SIZE), name="obj_attn_feat_ff_dense_1")(final_attn)
    final_obj_feat = TimeDistributed(BatchNorm(), name='obj_attn_feat_ff_bn_1')(final_obj_feat, training=config.TRAIN_BN)
    final_obj_feat = Activation('relu')(final_obj_feat)

    return final_obj_feat
