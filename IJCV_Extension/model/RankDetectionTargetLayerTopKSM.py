from tensorflow.keras.layers import *
import tensorflow as tf
from model import utils
import numpy as np


class RankDetectionTargetLayerTopKSM(Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TOP_K_DETECTION, (y1, x1, y2, x2)] in normalized
          coordinates

    target_deltas: [batch, TOP_K_DETECTION, NUM_CLASSES,
                    (dy, dx, log(dh), log(dw), class_id)]
                   Class-specific bbox refinements.
    target_ranks: [batch, TOP_K_DETECTION]. Integer class IDs.


    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, config, **kwargs):
        super(RankDetectionTargetLayerTopKSM, self).__init__(**kwargs)
        self.config = config

    def get_config(self):
        config = super(RankDetectionTargetLayerTopKSM, self).get_config()
        config["config"] = self.config.to_dict()
        return config

    def call(self, inputs):
        proposals = inputs[0]   # Top K Detected Objects - from DetectionLayer
        obj_feat = inputs[1]
        gt_boxes = inputs[2]
        gt_ranks = inputs[3]

        # Slice the batch and run a graph for each slice
        # TODO: Rename target_bbox to target_deltas for clarity
        names = ["rois_top_k", "obj_feat_top_k", "target_bbox_top_k", "target_ranks_top_k", "target_spatial_masks_top_k"]
        outputs = utils.batch_slice(
            [proposals, obj_feat, gt_boxes, gt_ranks],
            lambda w, x, y, z: detection_targets_graph(
                w, x, y, z, self.config),
            self.config.IMAGES_PER_GPU, names=names)

        # Ensure shape of object feature
        outputs[1] = tf.reshape(outputs[1], [self.config.BATCH_SIZE, self.config.TOP_K_DETECTION, 1, 1, 1024])
        outputs[4] = tf.reshape(outputs[4], [self.config.BATCH_SIZE, self.config.TOP_K_DETECTION, 32, 32, 1])

        return outputs

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TOP_K_DETECTION, 4),  # rois
            (None, self.config.TOP_K_DETECTION, 1, 1, 1024),    # object features
            (None, self.config.TOP_K_DETECTION, 4),  # deltas
            (None, self.config.TOP_K_DETECTION),  # ranks
            (None, self.config.TOP_K_DETECTION, 32, 32, 1)  # spatial masks
        ]

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None, None]


def overlaps_graph(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    """
    # 1. Tile boxes2 and repeat boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeat() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(input=boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(input=boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(input=boxes1)[0], tf.shape(input=boxes2)[0]])
    return overlaps


def detection_targets_graph(proposals, object_features, gt_boxes, gt_ranks, config):
    """Generates detection targets for one image. Subsamples proposals and
    generates target class IDs, bounding box deltas, and masks for each.

    Inputs:
    proposals: [N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [MAX_GT_INSTANCES] int class IDs
    gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized coordinates.
    gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized coordinates
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs. Zero padded.
    deltas: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
            Class-specific bbox refinements.
    masks: [TRAIN_ROIS_PER_IMAGE, height, width). Masks cropped to bbox
           boundaries and resized to neural network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """
    # Assertions
    asserts = [
        tf.Assert(tf.greater(tf.shape(input=proposals)[0], 0), [proposals],
                  name="roi_assertion"),
    ]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    # Remove zero padding
    proposals, _ = trim_zeros_graph(proposals, name="trim_proposals")
    gt_boxes, non_zeros = trim_zeros_graph(gt_boxes, name="trim_gt_boxes")
    gt_ranks = tf.boolean_mask(tensor=gt_ranks, mask=non_zeros, name="trim_gt_ranks")

    obj_feat = tf.squeeze(object_features, axis=1)
    obj_feat = tf.squeeze(obj_feat, axis=1)

    obj_feat = tf.gather(obj_feat, tf.compat.v1.where(non_zeros)[:, 0], axis=0, name="trim_obj_feat")

    # Compute overlaps matrix [proposals, gt_boxes]
    overlaps = overlaps_graph(proposals, gt_boxes)

    # Determine positive and negative ROIs
    roi_iou_max = tf.reduce_max(input_tensor=overlaps, axis=1)
    # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
    positive_roi_bool = (roi_iou_max >= 0.5)
    positive_indices = tf.compat.v1.where(positive_roi_bool)[:, 0]
    # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
    negative_indices = tf.compat.v1.where(roi_iou_max < 0.5)[:, 0]

    # Gather selected ROIs
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    positive_obj_feat = tf.gather(obj_feat, positive_indices)
    negative_obj_feat = tf.gather(obj_feat, negative_indices)

    # Assign positive ROIs to GT boxes.
    positive_overlaps = tf.gather(overlaps, positive_indices)
    roi_gt_box_assignment = tf.cond(
        pred=tf.greater(tf.shape(input=positive_overlaps)[1], 0),
        true_fn=lambda: tf.argmax(input=positive_overlaps, axis=1),
        false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
    )
    roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
    roi_gt_ranks = tf.gather(gt_ranks, roi_gt_box_assignment)

    # Compute bbox refinement for positive ROIs
    deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
    deltas /= config.BBOX_STD_DEV

    # Append negative ROIs and pad bbox deltas and masks that
    # are not used for negative ROIs with zeros.
    rois = tf.concat([positive_rois, negative_rois], axis=0)
    N = tf.shape(input=negative_rois)[0]
    P = tf.maximum(config.TOP_K_DETECTION - tf.shape(input=rois)[0], 0)
    rois = tf.pad(tensor=rois, paddings=[(0, P), (0, 0)])
    roi_gt_boxes = tf.pad(tensor=roi_gt_boxes, paddings=[(0, N + P), (0, 0)])
    deltas = tf.pad(tensor=deltas, paddings=[(0, N + P), (0, 0)])

    roi_gt_ranks = tf.pad(tensor=roi_gt_ranks, paddings=[(0, N + P)], constant_values=0)

    obj_feat = tf.concat([positive_obj_feat, negative_obj_feat], axis=0)
    obj_feat = tf.pad(tensor=obj_feat, paddings=[(0, P), (0, 0)])

    obj_feat = tf.expand_dims(obj_feat, axis=1)
    obj_feat = tf.expand_dims(obj_feat, axis=1)

    # ------------------------------------------------------

    spatial_feat = generate_spatial_feat(roi_gt_boxes, config)

    return rois, obj_feat, deltas, roi_gt_ranks, spatial_feat


def generate_spatial_feat(rois, config):
    original_image_shape = (480, 640, 3)
    image_shape = (1024, 1024, 3)
    window = np.array([128, 0, 896, 1024])

    num_rois = config.TOP_K_DETECTION

    # Translate normalized coordinates in the resized image to pixel
    # coordinates in the original image before resizing
    window = utils.norm_boxes(window, image_shape[:2])

    wy1, wx1, wy2, wx2 = window
    shift = np.array([wy1, wx1, wy1, wx1])
    wh = wy2 - wy1  # window height
    ww = wx2 - wx1  # window width
    scale = np.array([wh, ww, wh, ww])

    shift = tf.convert_to_tensor(shift)
    scale = tf.convert_to_tensor(scale)

    # Convert boxes to normalized coordinates on the window
    boxes = tf.divide(rois - shift, scale)

    # Convert boxes to pixel coordinates on the original image
    # boxes = utils.denorm_boxes(boxes, original_image_shape[:2])
    h, w = original_image_shape[:2]
    scale = np.array([h - 1, w - 1, h - 1, w - 1])
    shift = np.array([0, 0, 1, 1])

    shift = tf.convert_to_tensor(shift)
    scale = tf.convert_to_tensor(scale)

    shift = tf.cast(shift, tf.float32)
    scale = tf.cast(scale, tf.float32)

    boxes = tf.round(tf.multiply(boxes, scale) + shift)
    boxes = tf.cast(boxes, tf.int32)

    # Filter out detections with zero area. Happens in early training when
    # network weights are still random
    # exclude_ix = tf.compat.v1.where(
    #     (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)
    # if exclude_ix.shape[0] > 0:
    #     boxes = np.delete(boxes, exclude_ix, axis=0)
    include_ix = tf.compat.v1.where(
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) > 0)
    include_ix = tf.cast(include_ix, tf.int32)

    updates = tf.gather(boxes, include_ix)
    updates = tf.transpose(updates, [0, 2, 1])

    shape = tf.constant([config.TRAIN_ROIS_PER_IMAGE, 4, 1])

    boxes = tf.scatter_nd(include_ix, updates, shape)
    boxes = tf.squeeze(boxes, axis=-1)

    # ----------
    # Boxes = [N, (y1, x1, y2, x2)]

    # Generate Spatial Masks
    spatial_masks_list = []
    for b in range(num_rois):
        box = boxes[b]
        y1, x1, y2, x2 = tf.split(box, 4)

        # Create mask
        mask = tf_bbox2mask(y1, x1, y2, x2, 480, 640)

        mask = tf.expand_dims(mask, axis=-1)

        # Resize to 32 x 32
        mask = tf.image.resize_with_pad(mask, target_height=32, target_width=32)

        spatial_masks_list.append(mask)

    spatial_masks = tf.stack(spatial_masks_list)

    return spatial_masks


# from https://gist.github.com/charliememory/34bc5a9a89c8eb849baae27edc90731e
def tf_bbox2mask(y1, x1, y2, x2, img_H, img_W):
    # Repeat for each row or column
    y1_transposed = tf.expand_dims(tf.tile(y1, [img_W]), 0)
    x1_transposed = tf.expand_dims(tf.tile(x1, [img_H]), 1)
    y2_transposed = tf.expand_dims(tf.tile(y2, [img_W]), 0)
    x2_transposed = tf.expand_dims(tf.tile(x2, [img_H]), 1)

    # Get the range grid
    range_row = tf.cast(tf.expand_dims(tf.range(0, img_H, 1), 1), tf.int32)
    range_col = tf.cast(tf.expand_dims(tf.range(0, img_W, 1), 0), tf.int32)

    # Generate boolean masks
    mask_y1 = tf.less(y1_transposed, range_row)
    mask_x1 = tf.less(x1_transposed, range_col)
    mask_y2 = tf.less(range_row, y2_transposed)
    mask_x2 = tf.less(range_col, x2_transposed)

    result = tf.cast(mask_y1, tf.float32) * tf.cast(mask_x1, tf.float32) * tf.cast(mask_y2, tf.float32) * tf.cast(mask_x2, tf.float32)
    return result


def trim_zeros_graph(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.
    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(input_tensor=tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(tensor=boxes, mask=non_zeros, name=name)
    return boxes, non_zeros
