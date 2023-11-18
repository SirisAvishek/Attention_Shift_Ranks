from Dataset import Dataset
import cv2
import numpy as np
from model import utils

WIDTH = 640
HEIGHT = 480

# Percentage of object pixels having predicted saliency value to consider as salient object
# for cases where object segments overlap each other
SEG_THRESHOLD = .5

RANKS_TO_SCORES = {
    255: 5,
    229: 4,
    204: 3,
    178: 2,
    153: 1,
    0: 0
}

GT_RANKS_TO_SCORES = {
    1.0: 5,
    0.9: 4,
    0.8: 3,
    0.7: 2,
    0.6: 1
}

RANK_VALS = [0.0, 255.0, 229.0, 204.0, 178.0, 153.0]
RANK_SCORE = [0, 5, 4, 3, 2, 1]

RANKS_ONLY = [255.0, 229.0, 204.0, 178.0, 153.0]


def load_saliency_map(path):
    # Load mask
    sal_map = cv2.imread(path, 1).astype(np.float32)

    # Need only one channel
    sal_map = sal_map[:, :, 0]

    # Normalize to 0-1
    sal_map /= 255.0

    return sal_map


def cal_ndcg(rel_true, rel_pred, p=None, form="linear"):
    """ Returns normalized Discounted Cumulative Gain
    Args:
        rel_true (1-D Array): relevance lists for particular user, (n_songs,)
        rel_pred (1-D Array): predicted relevance lists, (n_pred,)
        p (int): particular rank position
        form (string): two types of nDCG formula, 'linear' or 'exponential'
    Returns:
        ndcg (float): normalized discounted cumulative gain score [0, 1]
    """
    rel_true = np.sort(rel_true)[::-1]

    p = min(len(rel_true), min(len(rel_pred), p))
    discount = 1 / (np.log2(np.arange(p) + 2))

    if form == "linear":
        idcg = np.sum(rel_true[:p] * discount)
        dcg = np.sum(rel_pred[:p] * discount)
    elif form == "exponential" or form == "exp":
        idcg = np.sum([2**x - 1 for x in rel_true[:p]] * discount)
        dcg = np.sum([2**x - 1 for x in rel_pred[:p]] * discount)
    else:
        raise ValueError("Only supported for two formula, 'linear' or 'exp'")

    return dcg / idcg


def calculate_ndcg(dataset, model_pred_path):
    print("Calculating NDCG...")

    # Load GT Rank
    gt_rank_order = dataset.gt_rank_orders

    ndcg_lin_list = []
    ndcg_exp_list = []

    num = len(dataset.img_ids)
    for i in range(num):
        # ********** Dataset information
        sal_obj_idx = dataset.sal_obj_idx_list[i]

        N = len(sal_obj_idx)

        # load seg data
        obj_seg = dataset.obj_seg[i]
        instance_masks = []
        instance_pix_count = []

        # Create mask for each salient object
        for s_i in range(len(sal_obj_idx)):
            sal_idx = sal_obj_idx[s_i]

            # Get corresponding segmentation data
            seg = obj_seg[sal_idx]

            # Binary mask of object segment
            mask = utils.get_obj_mask(seg, HEIGHT, WIDTH)

            # Count number of pixels of object segment
            pix_count = mask.sum()

            instance_masks.append(mask)
            instance_pix_count.append(pix_count)

        # ********** Load Predicted Rank
        pred_data_path = model_pred_path + dataset.img_ids[i] + ".png"

        pred_sal_map = cv2.imread(pred_data_path)[:, :, 0]

        # Get corresponding predicted rank for each gt salient objects
        pred_ranks = []

        # Create mask for each salient object
        for s_i in range(len(instance_masks)):
            gt_seg_mask = instance_masks[s_i]
            gt_pix_count = instance_pix_count[s_i]

            pred_seg = np.where(gt_seg_mask == 1, pred_sal_map, 0)

            r = np.sum(pred_seg) / gt_pix_count

            pred_ranks.append(r)

        # Assign Ranks to Predicted
        sorted_pred_rank_list = sorted([(e, i) for i, e in enumerate(pred_ranks)], reverse=True)

        # New Predicted Rank Vals
        for n in range(len(sorted_pred_rank_list)):
            v, idx = sorted_pred_rank_list[n]

            r = RANKS_ONLY[n]

            # Assign new Rank Value if there is value
            if v > 0:
                pred_ranks[idx] = r

        # ********** Load GT Rank
        gt_rank_order_list = gt_rank_order[i]

        # Get Gt Rank Order of salient objects
        gt_ranks = []
        for j in range(N):
            s_idx = sal_obj_idx[j]
            gt_r = gt_rank_order_list[s_idx]
            gt_ranks.append(gt_r)

        # **************************************************

        # Order GT Ranks
        sorted_gt_rank_list = sorted([(e, i) for i, e in enumerate(gt_ranks)], reverse=True)

        # Create GT Scores - i.e. [5] * #GT_Objects
        gt_scores = [5] * len(gt_ranks)

        # Order Predicted Ranks according to the order of GT Ranks sorted
        sorted_pred_ranks = [pred_ranks[idx] for _, idx in sorted_gt_rank_list]

        # Normalize ranks for calculating rank differences - convert GT/Predicted Ranks into scores of same range (1-5)
        gt_vals = [GT_RANKS_TO_SCORES[x] for x, _ in sorted_gt_rank_list]

        pred_vals = [RANKS_TO_SCORES[x] for x in sorted_pred_ranks]

        # Diff
        diff = [abs(g_v - p_v) for g_v, p_v in zip(gt_vals, pred_vals)]

        # Predicted Ranks Scores
        pred_scores = [(5 - d) for d in diff]

        # Calculate NDCG
        ndcg_lin = cal_ndcg(gt_scores, pred_scores, len(gt_scores), "linear")
        ndcg_exp = cal_ndcg(gt_scores, pred_scores, len(gt_scores), "exp")

        ndcg_lin_list.append(ndcg_lin)
        ndcg_exp_list.append(ndcg_exp)

    avg_ndcg_lin = sum(ndcg_lin_list) / len(ndcg_lin_list)
    avg_ndcg_exp = sum(ndcg_exp_list) / len(ndcg_exp_list)

    return avg_ndcg_lin, avg_ndcg_exp


if __name__ == '__main__':
    print("Evaluate")

    # TODO: Change Here
    DATASET_ROOT = "...."   # Change to your location
    data_split = "test"
    dataset = Dataset(DATASET_ROOT, data_split)

    # TODO: Change Here
    model_pred_data_path = "...."   # Change to your location

    ####################################################

    _ndcg_lin, _ndcg_exp = calculate_ndcg(dataset, model_pred_data_path)

    print("\n----------------------------------------------------------")
    print("nDCG (lin): ", _ndcg_lin)
    print("nDCG (exp): ", _ndcg_exp)

