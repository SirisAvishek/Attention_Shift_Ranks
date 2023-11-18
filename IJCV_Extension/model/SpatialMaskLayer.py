from tensorflow.keras.layers import *
import tensorflow as tf
from model import utils
import numpy as np


class SpatialMaskLayer(Layer):
    def __init__(self, config, **kwargs):
        super(SpatialMaskLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        names = ["spatial_masks"]
        outputs = utils.batch_slice(
            [inputs],
            lambda x: generate_spatial_mask(x, self.config),
            self.config.BATCH_SIZE, names=names)

        outputs = tf.reshape(outputs, (self.config.BATCH_SIZE, self.config.TOP_K_DETECTION, 32, 32, 1))

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.config.TOP_K_DETECTION, 32, 32, 1)


def generate_spatial_mask(detections, config):
    original_image_shape = (480, 640, 3)
    image_shape = (1024, 1024, 3)
    window = np.array([128, 0, 896, 1024])

    # ----------

    # How many detections do we have?
    # Detections array is padded with zeros. Find the first class_id == 0.
    zero_ix = tf.compat.v1.where(detections[:, 4] == 0)[0]
    N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

    # Extract boxes, class_ids, scores
    boxes = detections[:N, :4]

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
    boxes = tf.divide(boxes - shift, scale)

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

    shape = tf.constant([config.DETECTION_MAX_INSTANCES, 4, 1])

    boxes = tf.scatter_nd(include_ix, updates, shape)
    boxes = tf.squeeze(boxes, axis=-1)

    # ----------
    # Boxes = [N, (y1, x1, y2, x2)]

    # Generate Spatial Masks
    spatial_masks_list = []
    for b in range(N):
        box = boxes[b]
        y1, x1, y2, x2 = tf.split(box, 4)

        # Create mask
        mask = tf_bbox2mask(y1, x1, y2, x2, 480, 640)

        mask = tf.expand_dims(mask, axis=-1)

        # Resize to 32 x 32
        mask = tf.image.resize_with_pad(mask, target_height=32, target_width=32)

        spatial_masks_list.append(mask)

    spatial_masks = tf.stack(spatial_masks_list)

    spatial_masks = tf.expand_dims(spatial_masks, axis=0)

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

