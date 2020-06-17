import numpy as np
import cv2
import json
import utils
import os
import skimage.color
import skimage.io
import skimage.transform
import pickle

MAX_OBJ_NUM = 30

# Top 5 salient objects should have vales > 0.5
SAL_VAL_THRESH = 0.5

"""
NOTES:
load_gt_rank_order(): 
     final_gt_rank: 0 = BG, 1 = Rank_1, 2 = Rank_2, 3 = Rank_3, 4 = Rank_4, 5 = Rank_5 
"""


class Dataset(object):
    def __init__(self, dataset_root, pre_proc_data_dir, data_split):
        self.dataset_root = dataset_root                    # Root folder of Dataset
        self.pre_proc_data_dir_root = pre_proc_data_dir     # Root folder of pre-processed data - Object Predictions
        self.data_split = data_split

        # Root folder for pre-processed data - i.e. object features
        self.obj_pre_proc_data_root = self.pre_proc_data_dir_root + "pre_process_feat/" + data_split + "/"

        self.load_dataset()

    def load_dataset(self):
        print("\nLoading Dataset...")

        image_file = self.data_split + "_images.txt"

        # Get list of image ids
        image_path = os.path.join(self.dataset_root, image_file)
        with open(image_path, "r") as f:
            image_names = [line.strip() for line in f.readlines()]

        self.img_ids = image_names
        print(self.img_ids)

        # Load Rank Order
        rank_order_root = self.dataset_root + "rank_order/" + self.data_split + "/"
        self.gt_rank_orders = self.load_rank_order_data(rank_order_root)

        # Load Object Data
        obj_seg_data_path = self.dataset_root + "obj_seg_data_" + self.data_split + ".json"

        self.obj_bboxes, self.obj_seg, self.sal_obj_idx_list, self.not_sal_obj_idx_list = self.load_object_seg_data(
            obj_seg_data_path)

    def load_rank_order_data(self, rank_order_root):
        rank_order_data_files = [f for f in os.listdir(rank_order_root)]

        gt_rank_orders = []
        for i in range(len(rank_order_data_files)):
            img_id = rank_order_data_files[i].split(".")[0]

            if img_id not in self.img_ids:
                continue

            p = rank_order_root + rank_order_data_files[i]

            with open(p, "r") as in_file:
                rank_data = json.load(in_file)

            rank_order = rank_data["rank_order"]

            gt_rank_orders.append(rank_order)

        return gt_rank_orders

    def load_object_seg_data(self, obj_data_path):
        with open(obj_data_path, "r") as f:
            data = json.load(f)

        obj_bbox = []
        obj_seg = []
        for i in range(len(data)):
            img_data = data[i]

            img_obj_data = img_data["object_data"]

            _img_obj_bbox = []
            _img_obj_seg = []
            for obj_data in img_obj_data:
                _img_obj_bbox.append(obj_data["bbox"])
                _img_obj_seg.append(obj_data["segmentation"])

            obj_bbox.append(_img_obj_bbox)
            obj_seg.append(_img_obj_seg)

        # Find N salient objects based on gt rank order
        _sal_obj_idx_list = []
        _not_sal_obj_idx_list = []
        # Create a set for defined salient objects
        for i in range(len(obj_bbox)):
            gt_ranks = np.array(self.gt_rank_orders[i])
            _idx_sal = np.where(gt_ranks > SAL_VAL_THRESH)[0].tolist()
            _sal_obj_idx_list.append(_idx_sal)

            _idx_not_sal = np.where(gt_ranks <= SAL_VAL_THRESH)[0].tolist()
            _not_sal_obj_idx_list.append(_idx_not_sal)

        return obj_bbox, obj_seg, _sal_obj_idx_list, _not_sal_obj_idx_list

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """

        # Load image
        p = self.dataset_root + "images/" + self.data_split + "/" + image_id + ".jpg"
        image = skimage.io.imread(p)

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_gt_mask(self, image_id):
        # Load mask
        p = self.dataset_root + "gt/" + self.data_split + "/" + image_id + ".png"
        og_gt_mask = cv2.imread(p, 1).astype(np.float32)

        # Need only one channel
        mask = og_gt_mask[:, :, 0]

        # Normalize to 0-1
        mask /= 255.0

        return np.array(mask)

    def load_obj_gt_mask(self, image_id):
        idx = self.img_ids.index(image_id)

        p = self.dataset_root + "gt/" + self.data_split + "/" + image_id + ".png"
        gt_mask = cv2.imread(p, 1).astype(np.float32)
        height, width = gt_mask.shape[:2]

        obj_seg = self.obj_seg[idx]
        sal_obj_idx = self.sal_obj_idx_list[idx]

        instance_masks = []
        class_ids = []

        # Create mask for each salient object
        for s_i in range(len(sal_obj_idx)):
            sal_idx = sal_obj_idx[s_i]

            # Get corresponding segmentation data
            seg = obj_seg[sal_idx]

            mask = utils.get_obj_mask(seg, height, width)

            instance_masks.append(mask)
            class_ids.append(1)

        obj_masks = np.stack(instance_masks, axis=2).astype(np.bool)
        class_ids = np.array(class_ids, dtype=np.int32)

        return obj_masks, class_ids

    def load_gt_rank_order(self, image_id):
        """
        Returns:
            final_gt_rank - [MAX_OBJ_NUM] array of shuffled gt rank, val = rank
                            i.e.  0 = BG, 1 = Rank_1, 2 = Rank_2, 3 = Rank_3, 4 = Rank_4, 5 = Rank_5
            non_sal_idx - list of randomly selected non-salient objects
                            if, No. of non-salient objects > number of non-salient objects required to fill MAX_OBJ_NUM
            final_obj_indices - shuffle order of the salient objects and non-salient objects indices,
                                remaining objects required to fill MAX_OBJ_NUM left are given an index of -99
            chosen_indices_order_list - the shuffle order index where the salient object and non-salient objects are
                                located - (contains only the selected/available salient & non-salient objects)
        """

        idx = self.img_ids.index(image_id)

        rank_orders = self.gt_rank_orders[idx]
        sal_obj_idx = self.sal_obj_idx_list[idx]
        not_sal_obj_idx = self.not_sal_obj_idx_list[idx]

        # If No. of non salient objects is > than the number of non-salient objects required to fill MAX_OBJ_NUM,
        # then, randomly pick non-salient objects
        non_sal_idx = not_sal_obj_idx
        remaining_num_obj = MAX_OBJ_NUM - len(sal_obj_idx)
        if len(not_sal_obj_idx) > remaining_num_obj:
            # select random non salient objects
            non_sal_idx = np.random.choice(not_sal_obj_idx, remaining_num_obj, replace=False).tolist()

        # Ranks of Salient Objects
        gt_rank_order_class = [0] * len(sal_obj_idx)
        for i in range(len(sal_obj_idx)):
            sal_id = sal_obj_idx[i]
            rank_order = rank_orders[sal_id]
            rank_cls = rank_order * 10
            rank_cls = 10 - rank_cls

            # class 0 = No rank/not salient
            rank_cls += 1
            gt_rank_order_class[i] = int(rank_cls)

        gt_rank_order = np.array(gt_rank_order_class, dtype=np.int32)

        # First N objects will be salient, rest are given 0 (rank)
        final_gt_rank = np.zeros(shape=(MAX_OBJ_NUM,), dtype=np.int32)
        final_gt_rank[:gt_rank_order.shape[0]] = gt_rank_order

        # Object Indices
        final_obj_indices = sal_obj_idx + non_sal_idx
        # Fill remaining with -1
        while len(final_obj_indices) < MAX_OBJ_NUM:
            final_obj_indices.append(-99)

        # Random Shuffle order
        shuffle_order = np.array(range(MAX_OBJ_NUM), dtype=np.int32)
        np.random.shuffle(shuffle_order)

        # Shuffle data
        final_gt_rank = np.take(final_gt_rank, shuffle_order)
        final_obj_indices = np.take(final_obj_indices, shuffle_order)

        # Index order of salient objects followed by non-salient object
        # in the shuffle list
        chosen_indices_order_list = []
        o_indices = sal_obj_idx + non_sal_idx
        for i in range(len(o_indices)):
            _idx = final_obj_indices.tolist().index(o_indices[i])
            chosen_indices_order_list.append(_idx)

        return final_gt_rank, non_sal_idx, final_obj_indices, chosen_indices_order_list

    def load_object_roi_masks(self, image_id, sel_non_sal_obj):
        image = self.load_image(image_id)
        idx = self.img_ids.index(image_id)

        obj_bbox = self.obj_bboxes[idx]
        sal_obj_idx = self.sal_obj_idx_list[idx]

        if len(obj_bbox) < 1:
            obj_masks = np.empty([0, 0, 0])
            return obj_masks

        rois = []
        # Get list of salient object bboxes
        for i in range(len(sal_obj_idx)):
            _idx = sal_obj_idx[i]
            obj = obj_bbox[_idx]
            rois.append(obj)

        # Get list of randomly selected non-salient object bboxes
        for i in range(len(sel_non_sal_obj)):
            _idx = sel_non_sal_obj[i]
            obj = obj_bbox[_idx]
            rois.append(obj)

        # Reference Mask
        image_shape = image.shape[:2]
        init_mask = np.zeros(shape=image_shape, dtype=np.int32)

        # Generate list of object masks from salient and randomly selected non salient objects if available
        obj_mask_instances = []
        for i in range(len(rois)):
            obj = rois[i]
            o_x1, o_y1, o_x2, o_y2 = obj

            obj_mask = init_mask.copy()
            obj_mask[o_y1:o_y2, o_x1:o_x2] = 1
            obj_mask_instances.append(obj_mask)

        obj_masks = np.stack(obj_mask_instances, axis=2)

        return obj_masks

    def get_image_info(self, image_id):
        img_path = self.dataset_root + "images/" + self.data_split + "/" + image_id + ".jpg"

        return img_path

    def load_obj_pre_proc_data(self, image_id):
        p = self.obj_pre_proc_data_root + image_id

        with open(p, "rb") as f:
            obj_data = pickle.load(f)

        return obj_data

    def load_gt_rank_and_obj_feat_with_pre_proc_data(self, image_id):
        p = self.obj_pre_proc_data_root + image_id

        with open(p, "rb") as f:
            obj_data = pickle.load(f)

        # obj_masks = obj_data["obj_masks"]
        obj_feat = obj_data["obj_feat"]
        shuffled_indices = obj_data["shuffled_indices"]
        sel_not_sal_obj_idx_list = obj_data["sel_not_sal_obj_idx_list"]
        chosen_obj_idx_order_list = obj_data["chosen_obj_idx_order_list"]
        P5 = obj_data["P5"]

        idx = self.img_ids.index(image_id)
        rank_orders = self.gt_rank_orders[idx]

        gt_ranks = []
        for i in range(len(shuffled_indices)):
            o_idx = shuffled_indices[i]

            if o_idx == -99:
                gt_ranks.append(0)  # Filled Object - non existent object
            else:
                rank_order = rank_orders[o_idx]

                # Check if salient object
                if rank_order > 0.5:
                    rank_cls = rank_order * 10
                    rank_cls = 10 - rank_cls

                    rank_cls += 1
                    gt_ranks.append(int(rank_cls))
                else:
                    gt_ranks.append(0)  # Non-salient Object

        gt_rank_order = np.array(gt_ranks, dtype=np.int32)

        return gt_rank_order, obj_feat, shuffled_indices, sel_not_sal_obj_idx_list, chosen_obj_idx_order_list, P5
