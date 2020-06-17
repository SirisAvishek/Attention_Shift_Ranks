import numpy as np
from fpn_network import model_utils
from fpn_network import utils


def load_inference_data_obj_feat(dataset, image_id, config):
    image = dataset.load_image(image_id)

    object_roi_masks = dataset.load_object_roi_masks(image_id)

    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    obj_mask = utils.resize_mask(object_roi_masks, scale, padding, crop)

    # bbox: [num_instances, (y1, x1, y2, x2)]
    obj_bbox = utils.extract_bboxes(obj_mask)

    # Normalize image
    image = model_utils.mold_image(image.astype(np.float32), config)

    # Active classes
    active_class_ids = np.ones([config.NUM_CLASSES], dtype=np.int32)
    img_id = image_id
    img_id = int(img_id[-12:])
    # Image meta data
    image_meta = model_utils.compose_image_meta(img_id, original_shape, image.shape,
                                                window, scale, active_class_ids)

    # Expand input dimensions to consider batch
    image = np.expand_dims(image, axis=0)
    image_meta = np.expand_dims(image_meta, axis=0)

    batch_obj_roi = np.zeros(shape=(config.SAL_OBJ_NUM, 4), dtype=np.int32)
    batch_obj_roi[:len(obj_bbox)] = obj_bbox
    batch_obj_roi = np.expand_dims(batch_obj_roi, axis=0)

    return [image, image_meta, batch_obj_roi]


def load_inference_data_obj_feat_gt(dataset, image_id, config):
    image = dataset.load_image(image_id)

    gt_ranks, sel_not_sal_obj_idx_list, shuffled_indices, chosen_obj_idx_order_list = dataset.load_gt_rank_order(image_id)

    object_roi_masks = dataset.load_object_roi_masks(image_id, sel_not_sal_obj_idx_list)

    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    obj_mask = utils.resize_mask(object_roi_masks, scale, padding, crop)

    # bbox: [num_instances, (y1, x1, y2, x2)]
    obj_bbox = utils.extract_bboxes(obj_mask)

    # *********************** FILL REST, SHUFFLE ORDER ***********************
    # order is in salient objects then non-salient objects
    batch_obj_roi = np.zeros(shape=(config.SAL_OBJ_NUM, 4), dtype=np.int32)
    for i in range(len(chosen_obj_idx_order_list)):
        _idx = chosen_obj_idx_order_list[i]
        batch_obj_roi[_idx] = obj_bbox[i]

    # Normalize image
    image = model_utils.mold_image(image.astype(np.float32), config)

    # Active classes
    active_class_ids = np.ones([config.NUM_CLASSES], dtype=np.int32)
    img_id = image_id
    img_id = int(img_id[-12:])
    # Image meta data
    image_meta = model_utils.compose_image_meta(img_id, original_shape, image.shape,
                                                window, scale, active_class_ids)

    # Expand input dimensions to consider batch
    image = np.expand_dims(image, axis=0)
    image_meta = np.expand_dims(image_meta, axis=0)
    batch_obj_roi = np.expand_dims(batch_obj_roi, axis=0)

    return [image, image_meta, batch_obj_roi], gt_ranks, sel_not_sal_obj_idx_list, shuffled_indices, chosen_obj_idx_order_list
