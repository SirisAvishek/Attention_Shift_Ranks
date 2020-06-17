from pycocotools import mask as maskUtils
import numpy as np


def get_obj_mask(seg_ann_data, height, width):
    """
    Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
    :return: binary mask (numpy 2D array)
    """
    if isinstance(seg_ann_data, list):
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(seg_ann_data, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(seg_ann_data['counts'], list):
        # uncompressed RLE
        rle = maskUtils.frPyObjects(seg_ann_data, height, width)
    else:
        # rle
        # rle = seg_ann_data['segmentation']
        rle = seg_ann_data

    m = maskUtils.decode(rle)

    return m


# Keep only salient objects that are considered salient by both maps
def get_usable_salient_objects_agreed(image_1_list, image_2_list):
    # Remove indices list
    rm_list = []
    for idx in range(len(image_1_list)):
        v = image_1_list[idx]
        v2 = image_2_list[idx]

        if v == 0 or v2 == 0:
            rm_list.append(idx)

    # Use indices list
    use_list = list(range(0, len(image_1_list)))
    use_list = list(np.delete(np.array(use_list), rm_list))

    # Remove the indices
    x = np.array(image_1_list)
    y = np.array(image_2_list)
    x = list(np.delete(x, rm_list))
    y = list(np.delete(y, rm_list))

    return x, y, use_list



