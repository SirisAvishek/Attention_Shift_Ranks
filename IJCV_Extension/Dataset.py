import numpy as np
import cv2
import json
from model import utils
import os
import skimage.color
import skimage.io
import skimage.transform

# Top 5 salient objects should have vales > 0.5
SAL_VAL_THRESH = 0.5

"""
NOTES:
load_gt_rank_order(): 
     final_gt_rank: 0 = BG, 1 = Rank_1, 2 = Rank_2, 3 = Rank_3, 4 = Rank_4, 5 = Rank_5 
"""


class Dataset(object):
    def __init__(self, dataset_root, data_split):
        self.dataset_root = dataset_root                    # Root folder of Dataset
        self.data_split = data_split

        self.num_classes = 2  # Background + Salient

        self.load_dataset()

    def load_dataset(self):
        print("\nLoading Dataset...")

        image_file = self.data_split + "_images.txt"

        # Get list of image ids
        image_path = os.path.join(self.dataset_root, image_file)
        with open(image_path, "r") as f:
            image_names = [line.strip() for line in f.readlines()]

        self.img_ids = image_names
        print("images: ", len(self.img_ids))

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
        rank_orders = self.gt_rank_orders[idx]

        instance_masks = []
        class_ids = []
        gt_ranks = []

        # Create mask for each salient object
        for s_i in range(len(sal_obj_idx)):
            sal_idx = sal_obj_idx[s_i]

            # Get corresponding segmentation data
            seg = obj_seg[sal_idx]

            # Get Mask
            mask = utils.get_obj_mask(seg, height, width)

            # Get Rank
            rank = rank_orders[sal_idx]

            # Check if salient object
            if rank > 0.5:
                rank_cls = rank * 10

                # ----- Top Ranks - 5, 4, 3, 2, 1 - bigger value to higher ranks
                rank_cls -= 5

                gt_ranks.append(int(rank_cls))
            else:
                gt_ranks.append(0)  # Non-salient Object

            instance_masks.append(mask)
            class_ids.append(1)

        obj_masks = np.stack(instance_masks, axis=2).astype(np.bool)
        class_ids = np.array(class_ids, dtype=np.int32)
        gt_rank_order = np.array(gt_ranks, dtype=np.int32)

        return obj_masks, class_ids, gt_rank_order

    def get_image_info(self, image_id):
        img_path = self.dataset_root + "images/" + self.data_split + "/" + image_id + ".jpg"

        return img_path
