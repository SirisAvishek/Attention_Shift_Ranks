from evaluation.DatasetTest import DatasetTest
from sklearn.metrics import mean_absolute_error
import cv2
import numpy as np
import pickle
import os
import utils
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

    print("\n")
    avg_mae = sum(mae_list) / len(mae_list)
    print("Average MAE Images = ", avg_mae)


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
        pred_mask[pred_mask > 0] = 1
        gt_mask[gt_mask > 0] = 1

        # Flatten masks
        gt_mask = gt_mask.flatten()
        pred_mask = pred_mask.flatten()

        mae = mean_absolute_error(gt_mask, pred_mask)

        mae_list.append(mae)

    print("\n")
    avg_mae = sum(mae_list) / len(mae_list)
    print("Average MAE Images (Binary Masks) = ", avg_mae)


def calculate_spr(dataset, model_pred_data_path, out_path):
    print("Calculating SOR...")

    # Load GT Rank
    gt_rank_order = dataset.gt_rank_orders

    spr_data = []

    num = len(dataset.img_ids)
    for i in range(num):
        # Image Id
        image_id = dataset.img_ids[i]

        print("\n")
        print(i + 1, " / ", num, " - ", image_id)

        # ********** Dataset information
        # idx = dataset.img_ids.index(image_id)
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

            # number of pixels with predicted values
            pred_pix_loc = np.where(pred_seg > 0)

            pred_pix_num = len(pred_pix_loc[0])

            # Get rank of object
            r = 0
            if pred_pix_num > int(gt_pix_count * SEG_THRESHOLD):

                vals = pred_seg[pred_pix_loc[0], pred_pix_loc[1]]

                mode = sc.mode(vals)[0][0]
                r = mode

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

        spr = None

        if len(gt_ranks) > 1:
            spr = sc.spearmanr(gt_ranks, pred_ranks)
        elif len(gt_ranks) == 1:
            spr = 1

        d = [image_id, spr, use_indices_list]
        spr_data.append(d)

    with open(out_path, "wb") as f:
        pickle.dump(spr_data, f)


def extract_spr_value(data_list):
    use_idx_list = []
    spr = []
    for i in range(len(data_list)):
        s = data_list[i][1]

        if s == 1:
            spr.append(s)
            use_idx_list.append(i)
        elif s and not np.isnan(s[0]):
            spr.append(s[0])
            use_idx_list.append(i)

    return spr, use_idx_list


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


def eval_spr(spr_data_path):
    with open(spr_data_path, "rb") as f:
        spr_all_data = pickle.load(f)

    spr_data, spr_use_idx = extract_spr_value(spr_all_data)

    pos_l = []
    neg_l = []
    for i in range(len(spr_data)):
        if spr_data[i] > 0:
            pos_l.append(spr_data[i])
        else:
            neg_l.append(spr_data[i])

    print("Positive SPR: ", pos_l)
    print("Negative SPR: ", neg_l)
    print("Positive SPR: ", len(pos_l))
    print("Negative SPR: ", len(neg_l))

    avg_spr = cal_avg_spr(spr_data)
    avg_spr_norm = get_norm_spr(avg_spr)

    print("\n----------------------------------------------------------")
    print("Data path: ", spr_data_path)
    print(len(spr_data), "/", len(spr_all_data), " - ", (len(spr_all_data) - len(spr_data)), "Images Not used")
    print("Average SPR Saliency: ", avg_spr)
    print("Average SPR Saliency Normalized: ", avg_spr_norm)


if __name__ == '__main__':
    print("Evaluate")

    DATASET_ROOT = "D:/Desktop/ASSR/"   # Change to your location
    PRE_PROC_DATA_ROOT = "D:/Desktop/ASSR_Data/"   # Change to your location
    data_split = "test"
    dataset = DatasetTest(DATASET_ROOT, PRE_PROC_DATA_ROOT, data_split, eval_spr=True)

    ####################################################
    map_path = "../saliency_maps/"

    # Calculate MAE
    eval_mae(dataset, map_path)
    eval_mae_binary_mask(dataset, map_path)

    ####################################################
    out_root = "../spr_data/"
    out_path = out_root + "spr_data"
    if not os.path.exists(out_root):
        os.makedirs(out_root)

    # Calculate SOR
    calculate_spr(dataset, map_path, out_path)

    eval_spr(out_path)
