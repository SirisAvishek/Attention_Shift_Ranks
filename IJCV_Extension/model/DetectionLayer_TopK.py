from tensorflow.keras.layers import *
from model import model_utils
from model import utils
import tensorflow as tf


class DetectionLayer(Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, mode, config=None, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.mode = mode
        self.config = config

    def get_config(self):
        config = super(DetectionLayer, self).get_config()
        config["config"] = self.config.to_dict()
        return config

    def call(self, inputs):
        rois = inputs[0]
        fpn_class = inputs[1]
        fpn_bbox = inputs[2]
        obj_feat = inputs[3]
        image_meta = inputs[4]

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        m = model_utils.parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = model_utils.norm_boxes_graph(m['window'], image_shape[:2])

        # Run detection refinement graph on each item in the batch
        detections_batch = utils.batch_slice(
            [rois, fpn_class, fpn_bbox, obj_feat, window],
            lambda x, y, w, o, z: refine_detections_graph(x, y, w, o, z, self.mode, self.config),
            self.config.IMAGES_PER_GPU)

        detections_batch[0] = tf.reshape(detections_batch[0], [self.config.BATCH_SIZE, self.config.TOP_K_DETECTION, 6])
        detections_batch[1] = tf.reshape(detections_batch[1], [self.config.BATCH_SIZE, self.config.TOP_K_DETECTION, 1, 1, 1024])

        return detections_batch

    def compute_output_shape(self, input_shape):
        return [
            (None, self.config.TOP_K_DETECTION, 6),
            (None, self.config.TOP_K_DETECTION, 1, 1, 1024)
        ]


def refine_detections_graph(rois, probs, deltas, obj_feature, window, mode, config):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in image coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [N, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # Class IDs per ROI
    class_ids = tf.argmax(input=probs, axis=1, output_type=tf.int32)

    # Class probability of the top class of each ROI
    num_rois = probs.shape[0]

    if mode == 'training':
        num_rois = config.TRAIN_ROIS_PER_IMAGE

    indices = tf.stack([tf.range(num_rois), class_ids], axis=1)    # For Training
    # indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)

    class_scores = tf.gather_nd(probs, indices)
    # Class-specific bounding box deltas
    deltas_specific = tf.gather_nd(deltas, indices)
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = model_utils.apply_box_deltas_graph(
        rois, deltas_specific * config.BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = model_utils.clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area

    # Filter out background boxes
    keep = tf.compat.v1.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if config.DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.compat.v1.where(class_scores >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(conf_keep, 0))
        keep = tf.sparse.to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois, keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.compat.v1.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
            tf.gather(pre_nms_rois, ixs),
            tf.gather(pre_nms_scores, ixs),
            max_output_size=config.TOP_K_DETECTION,
            iou_threshold=config.DETECTION_NMS_THRESHOLD)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = config.TOP_K_DETECTION - tf.shape(input=class_keep)[0]
        class_keep = tf.pad(tensor=class_keep, paddings=[(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([config.TOP_K_DETECTION])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                         dtype=tf.int64)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.compat.v1.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                tf.expand_dims(nms_keep, 0))
    keep = tf.sparse.to_dense(keep)[0]
    # Keep top detections
    roi_count = config.TOP_K_DETECTION
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(input=class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.dtypes.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
    ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = config.TOP_K_DETECTION - tf.shape(input=detections)[0]
    detections = tf.pad(tensor=detections, paddings=[(0, gap), (0, 0)], mode="CONSTANT")

    # -------------------------------

    ref_obj_feat = tf.gather(obj_feature, keep)

    ref_obj_feat = tf.squeeze(ref_obj_feat, axis=1)
    ref_obj_feat = tf.squeeze(ref_obj_feat, axis=1)

    ref_obj_feat = tf.pad(tensor=ref_obj_feat, paddings=[(0, gap), (0, 0)], mode="CONSTANT")

    ref_obj_feat = tf.expand_dims(ref_obj_feat, axis=1)
    ref_obj_feat = tf.expand_dims(ref_obj_feat, axis=1)

    return detections, ref_obj_feat
