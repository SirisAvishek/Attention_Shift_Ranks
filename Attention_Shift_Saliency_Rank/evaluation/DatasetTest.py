import numpy as np
import cv2
import json
import os
import skimage.color
import skimage.io
import skimage.transform
import pickle

MAX_OBJ_NUM = 30

IOU_THRESH = 0.5

# Top 5 salient objects should have vales > 0.5
SAL_VAL_THRESH = 0.5

OBJ_THRESH = 0.5


"""
NOTES:
load_gt_rank_order(): 
     final_gt_rank: 0 = BG, 1 = Rank_1, 2 = Rank_2, 3 = Rank_3, 4 = Rank_4, 5 = Rank_5 
"""


class DatasetTest(object):
    def __init__(self, dataset_root, pre_proc_data_dir, data_split, eval_spr=None):
        self.dataset_root = dataset_root                    # Root folder of Dataset
        self.pre_proc_data_dir_root = pre_proc_data_dir     # Root folder of pre-processed data - Object Predictions
        self.data_split = data_split

        self.load_dataset()
        self.load_obj_data()

        self.eval_spr = eval_spr
        if eval_spr:
            rank_order_root = self.dataset_root + "rank_order/" + self.data_split + "/"
            self.gt_rank_orders = self.load_rank_order_data(rank_order_root)

            obj_seg_data_path = self.dataset_root + "obj_seg_data_" + self.data_split + ".json"
            self.obj_bboxes, self.obj_seg, self.sal_obj_idx_list, self.not_sal_obj_idx_list = self.load_object_seg_data(
                obj_seg_data_path)

    def load_dataset(self):
        print("\nLoading Dataset...")

        image_file = self.data_split + "_images.txt"

        # Get list of image ids
        image_path = os.path.join(self.dataset_root, image_file)
        with open(image_path, "r") as f:
            image_names = [line.strip() for line in f.readlines()]

        self.img_ids = image_names
        print(self.img_ids)

    def load_obj_data(self):
        data_path = self.pre_proc_data_dir_root + "object_detection_feat/object_detection_test_images.pkl"

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.image_id = []
        self.rois = []
        # self.class_ids = []
        # self.scores = []

        for i in range(len(data)):
            d = data[i]
            image_id = d["image_id"]
            rois = d["rois"]
            # class_ids = d["class_ids"]
            scores = d["scores"]

            num_good_objects = sum(s > OBJ_THRESH for s in scores)

            keep_point = num_good_objects
            if keep_point > MAX_OBJ_NUM:
                keep_point = MAX_OBJ_NUM

            self.image_id.append(image_id)
            self.rois.append(rois[:keep_point])
            # self.class_ids.append(class_ids[:keep_point])
            # self.scores.append(scores[:keep_point])

        assert self.img_ids == self.image_id

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

    def load_object_roi_masks(self, image_id):
        image = self.load_image(image_id)
        idx = self.img_ids.index(image_id)

        rois = self.rois[idx]

        if len(rois) < 1:
            obj_masks = np.empty([0, 0, 0])
            return obj_masks

        # Reference Mask
        image_shape = image.shape[:2]
        init_mask = np.zeros(shape=image_shape, dtype=np.int32)

        # Generate list of object masks from salient and randomly selected non salient objects if available
        obj_mask_instances = []
        for i in range(len(rois)):
            obj = rois[i]
            # o_x1, o_y1, o_x2, o_y2 = obj
            o_y1, o_x1, o_y2, o_x2 = obj    # original coco format

            obj_mask = init_mask.copy()
            obj_mask[o_y1:o_y2, o_x1:o_x2] = 1
            obj_mask_instances.append(obj_mask)

        obj_masks = np.stack(obj_mask_instances, axis=2).astype(np.bool)

        return obj_masks

    def get_image_info(self, image_id):
        # idx = self.img_ids.index(image_id)
        img_path = self.dataset_root + "images/" + self.data_split + "/" + image_id + ".jpg"

        return img_path

    def load_obj_pre_proc_data(self, image_id):
        p = self. pre_proc_data_dir_root + "object_detection_feat/" + self.data_split + "/" + image_id

        with open(p, "rb") as f:
            obj_data = pickle.load(f)

        return obj_data
