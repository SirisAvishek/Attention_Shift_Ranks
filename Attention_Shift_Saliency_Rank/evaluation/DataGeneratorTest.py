import numpy as np
from fpn_network import utils


def load_inference_data(dataset, image_id, config):

    # Load GT data
    pre_proc_data = dataset.load_obj_pre_proc_data(image_id)

    obj_feat = pre_proc_data["obj_feat"]
    p5_feat = pre_proc_data["P5"]

    # Load Object Spatial Mask
    object_roi_masks = dataset.load_object_roi_masks(image_id)

    # For 32 x 32 image size
    scale = 0.05
    padding = [(4, 4), (0, 0), (0, 0)]
    crop = None
    obj_spatial_masks = utils.resize_mask(object_roi_masks, scale, padding, crop)

    # Transpose and add dimension
    # [32, 32, N] -> [N, 32, 32, 1]
    obj_spatial_masks = np.expand_dims(np.transpose(obj_spatial_masks, [2, 0, 1]), -1)

    # fill rest with 0
    batch_obj_spatial_masks = np.zeros(shape=(config.SAL_OBJ_NUM, 32, 32, 1), dtype=np.float32)

    batch_obj_spatial_masks[:obj_spatial_masks.shape[0]] = obj_spatial_masks

    batch_obj_spatial_masks = np.expand_dims(batch_obj_spatial_masks, axis=0)

    return [obj_feat, batch_obj_spatial_masks, p5_feat]
