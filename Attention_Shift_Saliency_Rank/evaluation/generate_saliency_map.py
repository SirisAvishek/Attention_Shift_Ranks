from evaluation.DatasetTest import DatasetTest
import numpy as np
import pickle
import os
from fpn_network import utils
from model.CustomConfigs import RankModelConfig
import cv2

RANK_VALS = [0.0, 255.0, 229.0, 204.0, 178.0, 153.0]
RANK_SCORE = [0, 5, 4, 3, 2, 1]


def load_data_single(data_path):

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    return data


if __name__ == '__main__':

    model_pred_data_path = "../predictions/"

    out_path = "../saliency_maps/"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    DATASET_ROOT = "D:/Desktop/ASSR/"   # Change to your location
    PRE_PROC_DATA_ROOT = "D:/Desktop/ASSR_Data/"   # Change to your location
    data_split = "test"
    dataset = DatasetTest(DATASET_ROOT, PRE_PROC_DATA_ROOT, data_split,)

    config = RankModelConfig()

    num = len(dataset.img_ids)
    for i in range(num):
        image_id = dataset.img_ids[i]
        print("\n", i + 1, " / ", num, " - ", image_id)

        # ********** Model Predictions
        data_path = model_pred_data_path + dataset.img_ids[i]
        pred = load_data_single(data_path)

        ranks = np.squeeze(pred, axis=0)

        # ********** Dataset information
        og_image = dataset.load_image(image_id)
        original_shape = og_image.shape

        # Pre-processed Object Mask
        pre_proc_data = dataset.load_obj_pre_proc_data(image_id)
        obj_masks = pre_proc_data["obj_masks"]
        obj_masks = np.squeeze(obj_masks, axis=0)

        # Get Original bounding box for resize
        object_roi_masks = dataset.load_object_roi_masks(image_id)
        obj_roi = utils.extract_bboxes(object_roi_masks)

        # ********** Process Predicted Masks
        # First get saliency values
        masks = np.array_split(obj_masks, 2, axis=-1)
        masks = masks[1]
        masks = np.squeeze(masks, axis=-1)

        # Remove "padded" data
        masks = masks[:obj_roi.shape[0]]
        ranks = ranks[:obj_roi.shape[0]]

        # Resize the predicted masks to original size and location based on their available bounding boxes
        # Resize masks to original image size and set boundary threshold.
        N = obj_roi.shape[0]
        full_masks = []
        for j in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[j], obj_roi[j], original_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1) \
            if full_masks else np.empty(original_shape[:2] + (0,))

        # ********** Process Predicted Ranks
        rank_cls = np.argmax(ranks, axis=-1)

        rank_cls = np.reshape(rank_cls, (-1, 1))

        rank_prob = np.take_along_axis(ranks, rank_cls, axis=1)

        np_rank_scores = np.take(RANK_SCORE, rank_cls)

        rank_scores = np_rank_scores + rank_prob

        rank_scores *= np_rank_scores

        rank_scores = np.reshape(rank_scores, (-1))

        sorted_rank_list = sorted([(e, i) for i, e in enumerate(rank_scores)], reverse=True)

        # ********** Generate Grey-scale Saliency Rank Map
        sal_map = np.zeros(shape=(480, 640))

        num_pred_ranks = len(sorted_rank_list)
        num_ranks = 5
        if num_pred_ranks > 5:
            num_ranks = 5
        else:
            num_ranks = num_pred_ranks

        for j in range(num_ranks):
            p_r = sorted_rank_list[j][0]
            idx = sorted_rank_list[j][1]

            # Break once we reach BG objects
            if not p_r > 0:
                break

            val = RANK_VALS[j+1]

            sal_map = np.where(full_masks[:, :, idx] == 1, val, sal_map)

        sal_mask = np.zeros(shape=(480, 640, 3))
        sal_mask[:, :, 0] = sal_map
        sal_mask[:, :, 1] = sal_map
        sal_mask[:, :, 2] = sal_map

        f = out_path + image_id + ".png"

        cv2.imwrite(f, sal_mask)
