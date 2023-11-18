from Dataset import Dataset
from sklearn.metrics import mean_absolute_error
import cv2
import numpy as np
from model import utils
import scipy.stats as sc

WIDTH = 640
HEIGHT = 480

# Percentage of object pixels having predicted saliency value to consider as salient object
# for cases where object segments overlap each other
SEG_THRESHOLD = .5


def load_saliency_map(path):
    # Load mask
    sal_map = cv2.imread(path, 1).astype(np.float32)

    # Need only one channel
    sal_map = sal_map[:, :, 0]

    # Normalize to 0-1
    sal_map /= 255.0

    return sal_map


def eval_mae(dataset, map_path):
    print("Calculating MAE...")

    mae_list = []

    num = len(dataset.img_ids)
    for i in range(num):
        image_id = dataset.img_ids[i]

        p = map_path + image_id + ".png"

        pred_mask = load_saliency_map(p)

        gt_mask = dataset.load_gt_mask(image_id)

        # Flatten masks
        gt_mask = gt_mask.flatten()
        pred_mask = pred_mask.flatten()

        mae = mean_absolute_error(gt_mask, pred_mask)

        mae_list.append(mae)

    avg_mae = sum(mae_list) / len(mae_list)

    return avg_mae


def eval_mae_binary_mask(dataset, map_path):
    print("Calculating MAE (Binary Saliency)...")

    mae_list = []

    num = len(dataset.img_ids)
    for i in range(num):
        image_id = dataset.img_ids[i]

        p = map_path + image_id + ".png"
        pred_mask = load_saliency_map(p)

        gt_mask = dataset.load_gt_mask(image_id)

        # Convert masks to binary
        gt_mask[gt_mask > 0] = 1
        pred_mask = np.round(pred_mask)

        # Flatten masks
        gt_mask = gt_mask.flatten()
        pred_mask = pred_mask.flatten()

        mae = mean_absolute_error(gt_mask, pred_mask)

        mae_list.append(mae)

    avg_mae = sum(mae_list) / len(mae_list)

    return avg_mae


def calculate_spr(dataset, model_pred_data_path):
    print("Calculating SOR...")

    # Load GT Rank
    gt_rank_order = dataset.gt_rank_orders

    spr_data = []

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
        pred_data_path = model_pred_data_path + dataset.img_ids[i] + ".png"

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

        # ********** Load GT Rank
        gt_rank_order_list = gt_rank_order[i]

        # Get Gt Rank Order of salient objects
        gt_ranks = []
        for j in range(N):
            s_idx = sal_obj_idx[j]
            gt_r = gt_rank_order_list[s_idx]
            gt_ranks.append(gt_r)

        # Remove objects with no saliency value in both list
        gt_ranks, pred_ranks, use_indices_list = \
            utils.get_usable_salient_objects_agreed(gt_ranks, pred_ranks)

        spr = 0

        # Require at least 2 rank prediction
        if len(gt_ranks) > 1:
            spr = sc.spearmanr(gt_ranks, pred_ranks)

        if spr and not np.isnan(spr[0]):
            spr_data.append(spr[0])

    return spr_data


def cal_avg_spr(data_list):
    spr = np.array(data_list)
    avg = np.average(spr)
    return avg


def get_norm_spr(spr_value):
    #       m - r_min
    # m -> ---------------- x (t_max - t_min) + t_min
    #       r_max - r_min
    #
    # m = measure value
    # r_min = min range of measurement
    # r_max = max range of measurement
    # t_min = min range of desired scale
    # t_max = max range of desired scale

    r_min = -1
    r_max = 1

    norm_spr = (spr_value - r_min) / (r_max - r_min)

    return norm_spr


def eval_spr(dataset, map_path):
    spr_data = calculate_spr(dataset, map_path)

    avg_spr = cal_avg_spr(spr_data)
    avg_spr_norm = get_norm_spr(avg_spr)

    return avg_spr, avg_spr_norm, len(spr_data), (len(dataset.img_ids) - len(spr_data))


if __name__ == '__main__':
    print("Evaluate")

    # TODO: Change Here
    DATASET_ROOT = "...."   # Change to your location
    data_split = "test"
    dataset = Dataset(DATASET_ROOT, data_split)

    # TODO: Change Here
    model_pred_data_path = "...."   # Change to your location

    ####################################################

    # Calculate MAE
    _mae = eval_mae(dataset, model_pred_data_path)
    _mae_norm = eval_mae_binary_mask(dataset, model_pred_data_path)

    # Calculate SOR
    _spr, _spr_norm, _img_use, _img_not_use = eval_spr(dataset, model_pred_data_path)

    print("\n----------------------------------------------------------")
    print("MAE: ", _mae)
    print("MAE (Binary): ", _mae_norm)

    print("\n")
    print(_img_use, "/", len(dataset.img_ids), " - ", _img_not_use, "Images Not used")
    print("Average SPR Saliency: ", _spr)
    print("Average SPR Saliency Normalized: ", _spr_norm)

