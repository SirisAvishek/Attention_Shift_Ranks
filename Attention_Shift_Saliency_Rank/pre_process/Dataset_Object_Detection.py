import os
import skimage.color
import skimage.io
import skimage.transform
import numpy as np
import pickle

MAX_OBJ_NUM = 30
OBJ_THRESH = 0.5


class Dataset(object):
    def __init__(self, dataset_root, data_split, obj_det_path=None):

        self.dataset_root = dataset_root
        self.data_split = data_split

        self.load_dataset()

        self.obj_det_path = obj_det_path
        if obj_det_path:
            self.load_object_data()

    def load_dataset(self):
        print("\nLoading Dataset...")

        image_file = self.data_split + "_images.txt"

        # Get list of image ids
        image_path = os.path.join(self.dataset_root, image_file)
        with open(image_path, "r") as f:
            image_names = [line.strip() for line in f.readlines()]

        self.img_ids = image_names
        print(self.img_ids)

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

    def load_object_data(self):
        data_path = self.obj_det_path + "object_detection_test_images.pkl"

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.image_id = []
        self.rois = []
        self.class_ids = []
        self.scores = []

        for i in range(len(data)):
            d = data[i]
            image_id = d["image_id"]
            rois = d["rois"]
            class_ids = d["class_ids"]
            scores = d["scores"]

            num_good_objects = sum(s > OBJ_THRESH for s in scores)

            keep_point = num_good_objects
            if keep_point > MAX_OBJ_NUM:
                keep_point = MAX_OBJ_NUM

            self.image_id.append(image_id)
            self.rois.append(rois[:keep_point])
            self.class_ids.append(class_ids[:keep_point])
            self.scores.append(scores[:keep_point])

        assert self.img_ids == self.image_id

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
            o_y1, o_x1, o_y2, o_x2 = obj

            obj_mask = init_mask.copy()
            obj_mask[o_y1:o_y2, o_x1:o_x2] = 1
            obj_mask_instances.append(obj_mask)

        obj_masks = np.stack(obj_mask_instances, axis=2).astype(np.bool)

        return obj_masks

    def get_image_info(self, image_id):
        img_path = self.dataset_root + "images/" + self.data_split + "/" + image_id + ".jpg"

        return img_path
