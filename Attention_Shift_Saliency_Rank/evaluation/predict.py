import os
import pickle
from evaluation.DatasetTest import DatasetTest
from model.CustomConfigs import InferenceConfig
from model.ASRNet import ASRNet
from evaluation import DataGeneratorTest
from model import Model_SAM_SMM

DATASET_ROOT = "D:/Desktop/ASSR/"   # Change to your location
PRE_PROC_DATA_ROOT = "D:/Desktop/ASSR_Data/"    # Change to your location

if __name__ == '__main__':
    weight_path = "../weights/" + ".h5"

    out_path = "../predictions/"

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    config = InferenceConfig()
    log_path = "logs/"
    mode = "inference"

    keras_model = Model_SAM_SMM.build_saliency_rank_model(config, mode)
    model_name = "Rank_Model_SAM_SMM"
    model = ASRNet(mode=mode, config=config, model_dir=log_path, keras_model=keras_model, model_name=model_name)

    # Load weights
    print("Loading weights ", weight_path)
    model.load_weights(weight_path, by_name=True)

    # ********** Create Datasets
    dataset = DatasetTest(DATASET_ROOT, PRE_PROC_DATA_ROOT, "test")

    # **************************************************
    print("Start Prediction...")

    predictions = []

    num = len(dataset.img_ids)
    for i in range(num):

        image_id = dataset.img_ids[i]
        print(i + 1, " / ", num, " - ", image_id)

        input_data = DataGeneratorTest.load_inference_data(dataset, image_id, config)

        result = model.detect(input_data, verbose=1)

        o_p = out_path + image_id
        with open(o_p, "wb") as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)



