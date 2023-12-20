import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

from model import model_utils
from model import RegionProposalNetwork
from model.ProposalLayer import ProposalLayer
from model.BatchNorm import BatchNorm

from model import SaliencyClassificationNetworks, Losses

from model.SpatialMaskLayer import SpatialMaskLayer

from model.DetectionTargetLayer import DetectionTargetLayer
from model.RankDetectionTargetLayerTopKSM import RankDetectionTargetLayerTopKSM

from model.DetectionLayer_TopK import DetectionLayer
from model.Modules import generate_backbone_features, get_anchors, fpn_classifier_graph, \
    mask_and_edge_network, object_spatial_mask_module, selective_attention_module

from model.ConstLayer import ConstLayer

tf.compat.v1.disable_eager_execution()


def build_saliency_rank_model(config, mode, train_mode=None):
    # resnet_arch = "resnet101"
    # use_stage_5 = True
    # train_bn = False, Using small batch size
    # USE_RPN_ROIS = True

    assert mode in ['training', 'inference']
    assert train_mode in ['pre', 'fine', 'inference']

    # # Image size must be dividable by 2 multiple times
    # h, w = config.IMAGE_SHAPE[:2]
    # if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
    #     raise Exception("Image size must be dividable by 2 at least 6 times "
    #                     "to avoid fractions when downscaling and upscaling."
    #                     "For example, use 256, 320, 384, 448, 512, ... etc. ")

    # *********************** INPUTS ***********************
    input_image = KL.Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")
    input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE], name="input_image_meta")

    if mode == "training":
        # RPN GT
        input_rpn_match = KL.Input(
            shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
        input_rpn_bbox = KL.Input(
            shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

        # Detection GT (class IDs, bounding boxes, and masks)
        # 1. GT Class IDs (zero padded)
        input_gt_class_ids = KL.Input(
            shape=[None], name="input_gt_class_ids", dtype=tf.int32)
        # 2. GT Boxes in pixels (zero padded)
        # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
        input_gt_boxes = KL.Input(
            shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
        # Normalize coordinates
        gt_boxes = KL.Lambda(lambda x: model_utils.norm_boxes_graph(
            x, K.shape(input_image)[1:3]))(input_gt_boxes)
        # 3. GT Masks (zero padded)
        # [batch, height, width, MAX_GT_INSTANCES]
        if config.USE_MINI_MASK:
            input_gt_masks = KL.Input(
                shape=[config.MINI_MASK_SHAPE[0],
                       config.MINI_MASK_SHAPE[1], None],
                name="input_gt_masks", dtype=bool)

            input_gt_edge_masks = KL.Input(
                shape=[config.MINI_MASK_SHAPE[0],
                       config.MINI_MASK_SHAPE[1], None],
                name="input_gt_edge_masks", dtype=bool)

        else:
            input_gt_masks = KL.Input(
                shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                name="input_gt_masks", dtype=bool)

            input_gt_edge_masks = KL.Input(
                shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                name="input_gt_edge_masks", dtype=bool)

        # ----------
        input_gt_ranks = KL.Input(shape=[None], name="input_gt_ranks")

    elif mode == "inference":
        # Anchors in normalized coordinates
        input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

    # *********************** BACKBONE FEATURES ***********************
    # Generate Backbone features
    # backbone_feat = [P2, P3, P4, P5]
    # rpn_features = [P2, P3, P4, P5, P6]
    # P2: (?, 256, 256, 256)
    # P3: (?, 128, 128, 256)
    # P4: (?, 64, 64, 256)
    # P5: (?, 32, 32, 256)
    backbone_feat, rpn_feature_maps, P5 = generate_backbone_features(input_image, config)

    # *********************** RPN ***********************
    # Anchors
    if mode == "training":
        anchors = get_anchors(config, config.IMAGE_SHAPE)
        # Duplicate across the batch dimension because Keras requires it
        anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
        # A hack to get around Keras's bad support for constants
        # anchors = Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        anchors = ConstLayer(anchors, name="anchors")(input_image)
    else:
        anchors = input_anchors

    # RPN Model
    rpn = RegionProposalNetwork.build_rpn_model(config.RPN_ANCHOR_STRIDE,
                                                len(config.RPN_ANCHOR_RATIOS), config.TOP_DOWN_PYRAMID_SIZE)
    # Loop through pyramid layers
    layer_outputs = []  # list of lists
    for p in rpn_feature_maps:
        layer_outputs.append(rpn([p]))
    # Concatenate layer outputs
    # Convert from list of lists of level outputs to list of lists
    # of outputs across levels.
    # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
    output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
    outputs = list(zip(*layer_outputs))
    outputs = [KL.Concatenate(axis=1, name=n)(list(o))
               for o, n in zip(outputs, output_names)]

    rpn_class_logits, rpn_class, rpn_bbox = outputs

    # Generate proposals
    # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
    # and zero padded.
    proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" \
        else config.POST_NMS_ROIS_INFERENCE
    rpn_rois = ProposalLayer(
        proposal_count=proposal_count,
        nms_threshold=config.RPN_NMS_THRESHOLD,
        name="ROI",
        config=config)([rpn_class, rpn_bbox, anchors])

    # *********************** MODEL ***********************
    if mode == "training":
        # Class ID mask to mark class IDs supported by the dataset the image
        # came from.
        active_class_ids = KL.Lambda(
            lambda x: model_utils.parse_image_meta_graph(x)["active_class_ids"]
        )(input_image_meta)

        if not config.USE_RPN_ROIS:
            # Ignore predicted ROIs and use ROIs provided as an input.
            input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                  name="input_roi", dtype=np.int32)
            # Normalize coordinates
            target_rois = KL.Lambda(lambda x: model_utils.norm_boxes_graph(x, K.shape(input_image)[1:3]))(input_rois)
        else:
            target_rois = rpn_rois

        # Generate detection targets
        # Subsamples proposals and generates target outputs for training
        # Note that proposal class IDs, gt_boxes, and gt_masks are zero
        # padded. Equally, returned rois and targets are zero padded.
        rois, target_roi_gt_boxes, target_class_ids, target_bbox, target_mask, target_ranks, target_edges = \
            DetectionTargetLayer(config, name="proposal_targets")([
                target_rois, input_gt_class_ids, gt_boxes, input_gt_masks, input_gt_ranks, input_gt_edge_masks])

        # *********************** Network Heads
        feat_pyr_net_class_logits, feat_pyr_net_class, feat_pyr_net_bbox, \
            object_feat = fpn_classifier_graph(rois, backbone_feat, input_image_meta,
                                               config.POOL_SIZE, config.NUM_CLASSES,
                                               train_bn=config.TRAIN_BN,
                                               fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

        obj_seg_masks, obj_edge_masks = mask_and_edge_network(rois, backbone_feat,
                                                              input_image_meta,
                                                              config.MASK_POOL_SIZE,
                                                              config.NUM_CLASSES,
                                                              train_bn=config.TRAIN_BN)

        # -------------------------
        # Detections
        # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
        # normalized coordinates
        detections, object_feat = DetectionLayer(mode, config, name="feat_pyr_net_detection")(
            [rois, feat_pyr_net_class, feat_pyr_net_bbox, object_feat, input_image_meta])

        detection_boxes = KL.Lambda(lambda x: x[..., :4], name="detection_boxes")(detections)

        # Generate detection targets
        # Subsamples detections and generates target outputs for training
        # Note that proposal gt_boxes, and gt_masks are zero
        # padded. Equally, returned rois and targets are zero padded.
        _, object_feat, _, target_ranks, spatial_masks = \
            RankDetectionTargetLayerTopKSM(config, name="detection_targets_top_k")([
                detection_boxes, object_feat, target_roi_gt_boxes, target_ranks])

        # *********************** PROCESS Image/P5 FEATURES ***********************
        img_feat = KL.Conv2D(config.BOTTLE_NECK_SIZE, (3, 3), name="img_feat_conv_1")(P5)
        img_feat = BatchNorm(name="img_feat_bn_1")(img_feat, training=config.TRAIN_BN)
        img_feat = KL.Activation('relu')(img_feat)

        img_feat = KL.GlobalAveragePooling2D()(img_feat)

        # -------------------------

        # -64 for concatenation with spatial masks
        reduc_dim = config.BOTTLE_NECK_SIZE - 64

        # Reduce dimension to BOTTLNECK
        obj_feature = KL.TimeDistributed(KL.Conv2D(reduc_dim, (1, 1)), name="obj_feat_reduce_conv1")(
            object_feat)
        obj_feature = KL.TimeDistributed(BatchNorm(), name='obj_feat_reduce_bn1')(obj_feature, training=config.TRAIN_BN)
        obj_feature = KL.Activation('relu')(obj_feature)

        obj_feature = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="obj_feat_squeeze")(obj_feature)

        # ------------------------- OBJECT SPATIAL MASK MODULE
        spatial_mask_feat = object_spatial_mask_module(spatial_masks, config)

        # ------------------------- POSITIONAL ENCODING
        obj_feature = KL.Concatenate()([obj_feature, spatial_mask_feat])

        # ------------------------- SELECTIVE ATTENTION MODULE
        sa_feat = selective_attention_module(config.NUM_ATTN_HEADS, obj_feature, img_feat, config)

        # ------------------------- FINAL OBJECT FEATURE

        dim = config.RANK_FEAT_SIZE

        dropout = 0.5

        # FC layer for reducing the attention features
        final_obj_feat = KL.TimeDistributed(KL.Dense(dim), name="obj_final_feat_dense_1")(sa_feat)
        final_obj_feat = KL.TimeDistributed(BatchNorm(), name='obj_final_feat_bn_1')(final_obj_feat, training=config.TRAIN_BN)
        final_obj_feat = KL.Activation('relu')(final_obj_feat)
        final_obj_feat = KL.TimeDistributed(KL.Dropout(dropout))(final_obj_feat)

        # ------------------------- SALIENT OBJECT RANK NETWORK
        # Perform Saliency Classification/Ranking
        sal_rank = SaliencyClassificationNetworks.sal_class_network(final_obj_feat, config)

        # *********************** Losses

        if train_mode == "fine":
            # RPN Losses
            rpn_class_loss = KL.Lambda(lambda x: Losses.rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: Losses.rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])

            # Salient Object Losses
            obj_sal_seg_class_loss = KL.Lambda(lambda x: Losses.feat_pyr_net_class_loss_graph(*x),
                                            name="obj_sal_seg_class_loss")(
                [target_class_ids, feat_pyr_net_class_logits, active_class_ids])
            obj_sal_seg_bbox_loss = KL.Lambda(lambda x: Losses.feat_pyr_net_bbox_loss_graph(*x),
                                           name="obj_sal_seg_bbox_loss")([target_bbox, target_class_ids, feat_pyr_net_bbox])

        # Salient Object Segmentation Mask Loss
        obj_sal_seg_mask_loss = KL.Lambda(lambda x: Losses.mask_loss_graph(*x), name="obj_sal_seg_mask_loss")(
            [target_mask, target_class_ids, obj_seg_masks])

        # Salient Object Edge Loss
        obj_sal_edge_loss = KL.Lambda(lambda x: Losses.crisp_boundary_loss_pos_contrib(*x), name="obj_sal_edge_loss")(
            [target_edges, target_class_ids, obj_edge_masks])

        # Saliency Rank Loss
        saL_rank_loss = KL.Lambda(lambda x: Losses.list_mle_loss(*x), name="sal_rank_loss")(
            [target_ranks, sal_rank])

        # *********************** FINAL MODEL
        # Model
        inputs = [input_image, input_image_meta,
                  input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks,
                  input_gt_ranks, input_gt_edge_masks]
        if not config.USE_RPN_ROIS:
            inputs.append(input_rois)

        if train_mode == "fine":
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       feat_pyr_net_class_logits, feat_pyr_net_class, feat_pyr_net_bbox, obj_seg_masks, obj_edge_masks,
                       rpn_rois,
                       rpn_class_loss, rpn_bbox_loss,
                       obj_sal_seg_class_loss, obj_sal_seg_bbox_loss,
                       obj_sal_seg_mask_loss,
                       saL_rank_loss, obj_sal_edge_loss]
        else:
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       feat_pyr_net_class_logits, feat_pyr_net_class, feat_pyr_net_bbox, obj_seg_masks, obj_edge_masks,
                       rpn_rois,
                       # rpn_class_loss, rpn_bbox_loss,
                       # obj_sal_seg_class_loss, obj_sal_seg_bbox_loss,
                       obj_sal_seg_mask_loss,
                       saL_rank_loss, obj_sal_edge_loss]

        model = KM.Model(inputs, outputs=outputs, name="sal_rank_model")
    else:
        # *********************** Network Heads
        # Proposal classifier and BBox regressor heads
        feat_pyr_net_class_logits, feat_pyr_net_class, feat_pyr_net_bbox, \
            object_feat = fpn_classifier_graph(rpn_rois, backbone_feat, input_image_meta,
                                               config.POOL_SIZE, config.NUM_CLASSES,
                                               train_bn=config.TRAIN_BN,
                                               fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

        # Detections
        # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
        # normalized coordinates
        # detections = DetectionLayer(config, name="feat_pyr_net_detection")(
        #     [rpn_rois, feat_pyr_net_class, feat_pyr_net_bbox, input_image_meta])
        detections, object_feat = DetectionLayer(mode, config, name="feat_pyr_net_detection")(
            [rpn_rois, feat_pyr_net_class, feat_pyr_net_bbox, object_feat, input_image_meta])

        # Create masks for detections
        detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)

        obj_seg_masks, obj_edge_masks = mask_and_edge_network(detection_boxes, backbone_feat,
                                                              input_image_meta,
                                                              config.MASK_POOL_SIZE,
                                                              config.NUM_CLASSES,
                                                              train_bn=config.TRAIN_BN)

        # *********************** PROCESS Image/P5 FEATURES ***********************
        img_feat = KL.Conv2D(config.BOTTLE_NECK_SIZE, (3, 3), name="img_feat_conv_1")(P5)
        img_feat = BatchNorm(name="img_feat_bn_1")(img_feat, training=config.TRAIN_BN)
        img_feat = KL.Activation('relu')(img_feat)

        img_feat = KL.GlobalAveragePooling2D()(img_feat)

        # -------------------------

        # -64 for concatenation with spatial masks
        reduc_dim = config.BOTTLE_NECK_SIZE - 64

        # Reduce dimension to BOTTLNECK
        obj_feature = KL.TimeDistributed(KL.Conv2D(reduc_dim, (1, 1)), name="obj_feat_reduce_conv1")(
            object_feat)
        obj_feature = KL.TimeDistributed(BatchNorm(), name='obj_feat_reduce_bn1')(obj_feature, training=config.TRAIN_BN)
        obj_feature = KL.Activation('relu')(obj_feature)

        obj_feature = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2), name="obj_feat_squeeze")(obj_feature)

        # ------------------------- OBJECT SPATIAL MASK MODULE
        spatial_masks = SpatialMaskLayer(config, name="spatial_mask_layer")(detections)

        spatial_mask_feat = object_spatial_mask_module(spatial_masks, config)

        # ------------------------- POSITIONAL ENCODING
        obj_feature = KL.Concatenate()([obj_feature, spatial_mask_feat])

        # ------------------------- SELECTIVE ATTENTION MODULE
        sa_feat = selective_attention_module(config.NUM_ATTN_HEADS, obj_feature, img_feat, config)

        # ------------------------- FINAL OBJECT FEATURE
        dim = config.RANK_FEAT_SIZE

        # FC layer for reducing the attention features
        final_obj_feat = KL.TimeDistributed(KL.Dense(dim), name="obj_final_feat_dense_1")(sa_feat)
        final_obj_feat = KL.TimeDistributed(BatchNorm(), name='obj_final_feat_bn_1')(final_obj_feat,
                                                                                     training=config.TRAIN_BN)
        final_obj_feat = KL.Activation('relu')(final_obj_feat)

        # ------------------------- SALIENT OBJECT RANK NETWORK
        # Perform Saliency Classification/Ranking
        sal_rank = SaliencyClassificationNetworks.sal_class_network(final_obj_feat, config)

        model = KM.Model(inputs=[input_image, input_image_meta, input_anchors],
                         outputs=[detections, feat_pyr_net_class, feat_pyr_net_bbox,
                                  obj_seg_masks, obj_edge_masks,
                                  rpn_rois, rpn_class, rpn_bbox,
                                  sal_rank
                                  ],
                         name="sal_rank_model")

    return model
