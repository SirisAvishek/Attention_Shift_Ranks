from CustomConfigs import InferenceConfig
from model.ASSRNet import ASSRNet
from Dataset import Dataset

from model import Model as Test_Model

import numpy as np
import os
import cv2

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

RANK_VALS = [0.0, 255.0, 229.0, 204.0, 178.0, 153.0]
RANK_SCORE = [0, 5, 4, 3, 2, 1]

# TODO: Change Here
DATASET_ROOT = "...."   # Change to your location


if __name__ == '__main__':
    # TODO: Change Here
    out_path = "...."   # Change to your location

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # TODO: Change Here
    weight_file = "...."   # Change to your location

    model_name = "ASSRNet"

    # ----------

    config = InferenceConfig()
    log_path = "logs/"
    mode = "inference"

    keras_model = Test_Model.build_saliency_rank_model(config, mode)
    model = ASSRNet(mode=mode, config=config, model_dir=log_path, keras_model=keras_model, model_name=model_name)

    # ********** Create Datasets
    dataset = Dataset(DATASET_ROOT, "test")

    # ----------

    # Load weights
    print("Loading weights ", weight_file)
    model.load_weights(weight_file, by_name=True)

    # **************************************************
    print("Start Prediction...")

    predictions = []

    num = len(dataset.img_ids)
    for i in range(num):
        image_id = dataset.img_ids[i]

        image = dataset.load_image(image_id)

        results = model.detect([image], verbose=0)[0]

        # ********** Model Predictions

        rois = results["rois"]
        class_ids = results["class_ids"]
        scores = results["scores"]
        masks = results["masks"]

        sal_ranks = results["sal_ranks"]

        sal_ranks = np.squeeze(sal_ranks, axis=-1).tolist()

        # # ********** Dataset information
        og_image = dataset.load_image(image_id)
        original_shape = og_image.shape

        # ********** Process Predicted Ranks
        # Sort the predictions - 1st = saliency classification, 2nd = rank score, 3rd = saliency classification prob
        sorted_pred_list = sorted([(i, e) for i, e in enumerate(sal_ranks)], key=lambda x: x[1], reverse=True)

        # ********** Generate Grey-scale Saliency Rank Map
        sal_map = np.zeros(shape=(480, 640))

        num_pred_ranks = len(sorted_pred_list)

        num_ranks = 5
        if num_pred_ranks < 5:
            num_ranks = num_pred_ranks

        for j in range(num_ranks):
            idx = sorted_pred_list[j][0]
            v = sorted_pred_list[j][1]

            val = RANK_VALS[j + 1]

            sal_map = np.where(masks[:, :, idx] == 1, val, sal_map)

        sal_mask = np.zeros(shape=(480, 640, 3))
        sal_mask[:, :, 0] = sal_map
        sal_mask[:, :, 1] = sal_map
        sal_mask[:, :, 2] = sal_map

        # -----

        # Save Image
        f = out_path + image_id + ".png"
        cv2.imwrite(f, sal_mask)



