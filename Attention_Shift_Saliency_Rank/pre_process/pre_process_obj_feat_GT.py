from model.CustomConfigs import RankModelConfig
import os
import pickle
from pre_process import Model_Obj_Feat
from pre_process import DataGenerator
from pre_process.PreProcNet import PreProcNet
from pre_process.Dataset import Dataset

DATASET_ROOT = "D:/Desktop/ASSR/"   # Change to your location
PRE_PROC_DATA_ROOT = "D:/Desktop/ASSR_Data/"    # Change to your location

if __name__ == '__main__':
    # add pre-trained weight path - backbone pre-trained on salient objects (binary, no rank)
    weight_path = ""

    # Run Script twice to generate pre-processed object features of GT objects for "train" and "val" data_splits
    data_split = "train"
    # data_split = "val"

    out_path = PRE_PROC_DATA_ROOT + "pre_process_feat/" + data_split + "/"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    mode = "inference"
    config = RankModelConfig()
    log_path = "logs/"

    keras_model = Model_Obj_Feat.build_obj_feat_model(config)
    model_name = "Obj_Feat_Net"

    model = PreProcNet(mode=mode, config=config, model_dir=log_path, keras_model=keras_model, model_name=model_name)

    # Load weights
    print("Loading weights ", weight_path)
    model.load_weights(weight_path, by_name=True)

    if mode == "inference":
        # ********** Create Datasets
        # Train/Val Dataset
        dataset = Dataset(DATASET_ROOT, data_split)

        predictions = []

        num = len(dataset.img_ids)
        for i in range(num):

            image_id = dataset.img_ids[i]
            print(i + 1, " / ", num, " - ", image_id)

            input_data, gt_ranks, sel_not_sal_obj_idx_list, shuffled_indices, chosen_obj_idx_order_list = DataGenerator.load_inference_data_obj_feat_gt(dataset, image_id, config)

            result = model.detect(input_data, verbose=1)
            result["gt_ranks"] = gt_ranks
            result["sel_not_sal_obj_idx_list"] = sel_not_sal_obj_idx_list
            result["shuffled_indices"] = shuffled_indices
            result["chosen_obj_idx_order_list"] = chosen_obj_idx_order_list

            o_p = out_path + image_id
            with open(o_p, "wb") as f:
                pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)


