from evaluation.DatasetTest import DatasetTest
import numpy as np
import pickle
import utils
from fpn_network import utils
from sklearn import metrics

WIDTH = 640
HEIGHT = 480

RANK_SCORE = [0, 5, 4, 3, 2, 1]


def load_data_single(data_path):
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    return data


def get_gt_masks(dataset, idx):
    # ********** Dataset information
    # idx = dataset.img_ids.index(image_id)
    sal_obj_idx = dataset.sal_obj_idx_list[idx]

    # load seg data
    obj_seg = dataset.obj_seg[idx]
    instance_masks = []

    # Create mask for each salient object
    for s_i in range(len(sal_obj_idx)):
        sal_idx = sal_obj_idx[s_i]

        # Get corresponding segmentation data
        seg = obj_seg[sal_idx]

        # Binary mask of object segment
        mask = utils.get_obj_mask(seg, HEIGHT, WIDTH)

        instance_masks.append(mask)

    gt_masks = np.stack(instance_masks)

    return gt_masks


def get_ranks_infer(rank_probs):
    rank_cls = np.argmax(rank_probs, axis=-1)

    ranks = rank_cls.copy()

    rank_cls = np.reshape(rank_cls, (-1, 1))

    rank_prob = np.take_along_axis(rank_probs, rank_cls, axis=1)

    np_rank_scores = np.take(RANK_SCORE, rank_cls)

    rank_scores = np_rank_scores + rank_prob

    rank_scores *= np_rank_scores

    rank_scores = np.reshape(rank_scores, (-1))

    sorted_rank_list = sorted([(e, i) for i, e in enumerate(rank_scores)], reverse=True)

    return sorted_rank_list, ranks


def get_predicted_masks(dataset, image_id, model_pred_data_path):
    # ********** Model Predictions
    data_path = model_pred_data_path + image_id
    pred = load_data_single(data_path)

    ranks = np.squeeze(pred, axis=0)

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
    sorted_ranks, supposed_ranks = get_ranks_infer(ranks)

    num_positive_pred = sum(r > 0 for r in supposed_ranks)

    num_pred_ranks = len(sorted_ranks)

    sal_seg_maps_list = []
    for j in range(num_pred_ranks):
        p_r = sorted_ranks[j][0]
        idx = sorted_ranks[j][1]

        # Break once we reach BG objects
        if not p_r > 0:
            break

        sal_map = full_masks[:, :, idx]
        sal_seg_maps_list.append(sal_map)

    predicted_masks = None
    has_prediction = False

    if num_positive_pred > 0:
        predicted_masks = np.stack(sal_seg_maps_list)
        has_prediction = True

    return predicted_masks, has_prediction


# @jit
def calc_iou(mask_a, mask_b):
    intersection = (mask_a+mask_b>=2).astype(np.float32).sum()
    iou = intersection / (mask_a+mask_b>=1).astype(np.float32).sum()
    return iou


# @jit
def calc_accu_recall(gt_masks, segmaps, iou_thresh, ious):
    num_TP = 0
    for i in range(len(gt_masks)):
        max_match = 0
        max_match_id = 0
        for j in range(len(segmaps)):
            if ious[i,j] > max_match:
                max_match = ious[i,j]
                max_match_id = j
        if max_match > iou_thresh:
            num_TP += 1
            ious[:, max_match_id] = 0

    recall = num_TP / len(gt_masks)
    accu = num_TP / len(segmaps)

    return recall, accu


def eval_image_ap(gt_masks, segmaps, iou_thresh):
    ious = np.zeros([100, 100])
    for i in range(len(gt_masks)):
        for j in range(len(segmaps)):
            ious[i, j] = calc_iou(gt_masks[i], segmaps[j])

    recall_accu = {}
    for i in range(segmaps.shape[0]):
        accu, recall = calc_accu_recall(gt_masks, segmaps[:i + 1], iou_thresh, ious.copy())
        if recall in recall_accu:
            if accu > recall_accu[recall]:
                recall_accu[recall] = accu
        else:
            recall_accu[recall] = accu

    recalls = list(recall_accu.keys())
    recalls.sort()
    accus = []
    for recall in recalls:
        accus.append(recall_accu[recall])
    accus = accus[:1] + accus
    recalls = [0] + recalls

    if segmaps.shape[0] > 0:
        ap = metrics.auc(recalls, accus)
    else:
        ap = 0

    return ap


def eval_map(dataset, model_pred_data_path):
    print("Calculating mAP...")

    aps_5_list = []
    aps_7_list = []

    num = len(dataset.img_ids)
    for i in range(num):
        # Image Id
        image_id = dataset.img_ids[i]

        print("\n")
        print(i + 1, " / ", num, " - ", image_id)

        # ********** Dataset information
        gt_masks = get_gt_masks(dataset, i)

        predicted_masks, has_pred = get_predicted_masks(dataset, image_id, model_pred_data_path)

        if has_pred:
            ap_5 = eval_image_ap(gt_masks, predicted_masks, 0.5)
            ap_7 = eval_image_ap(gt_masks, predicted_masks, 0.7)
        else:
            ap_5 = 0
            ap_7 = 0

        aps_5_list.append(ap_5)
        aps_7_list.append(ap_7)

    map_05 = sum(aps_5_list) / len(aps_5_list)
    map_07 = sum(aps_7_list) / len(aps_7_list)

    print("\n----------------------------------------------------------")
    print("mAP-05: ", map_05)
    print("mAP-07: ", map_07)


if __name__ == '__main__':
    print("Evaluate")

    DATASET_ROOT = "D:/Desktop/ASSR/"   # Change to your location
    PRE_PROC_DATA_ROOT = "D:/Desktop/ASSR_Data/"   # Change to your location
    data_split = "test"
    dataset = DatasetTest(DATASET_ROOT, PRE_PROC_DATA_ROOT, data_split, eval_spr=True)

    ####################################################
    model_pred_data_path = "../predictions/"

    # Calculate mAP
    eval_map(dataset, model_pred_data_path)
