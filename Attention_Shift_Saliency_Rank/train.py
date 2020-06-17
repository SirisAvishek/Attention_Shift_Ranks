from Dataset import Dataset
from model.CustomConfigs import RankModelConfig
from model.ASRNet import ASRNet
import DataGenerator
from model import Model_SAM_SMM

# Path to dataset
DATASET_ROOT = "D:/Desktop/ASSR/"   # Change to your location

# Path to pre-processed data - object features
PRE_PROC_DATA_ROOT = "D:/Desktop/ASSR_Data/"    # Change to your location

if __name__ == '__main__':
    weight_path = ""    # add pre-trained weight path

    command = "train"
    config = RankModelConfig()
    log_path = "logs/"
    mode = "training"

    print("Loading Rank Model")
    keras_model = Model_SAM_SMM.build_saliency_rank_model(config, mode)
    model_name = "Rank_Model_SAM_SMM"
    model = ASRNet(mode=mode, config=config, model_dir=log_path, keras_model=keras_model, model_name=model_name)

    # Load weights
    print("Loading weights ", weight_path)
    model.load_weights(weight_path, by_name=True)

    # Train/Evaluate Model
    if command == "train":
        print("Start Training...")

        # ********** Create Datasets
        # Train Dataset
        train_dataset = Dataset(DATASET_ROOT, PRE_PROC_DATA_ROOT, "train")

        # Val Dataset
        val_dataset = Dataset(DATASET_ROOT, PRE_PROC_DATA_ROOT, "val")

        # ********** Parameters
        # Image Augmentation
        # Right/Left flip 50% of the time
        # augmentation = imgaug.augmenters.Fliplr(0.5)
        augmentation = None

        # ********** Create Data generators
        train_generator = DataGenerator.data_generator(train_dataset, config, shuffle=True,
                                                       augmentation=augmentation,
                                                       batch_size=config.BATCH_NUM)
        val_generator = DataGenerator.data_generator(val_dataset, config, shuffle=True,
                                                     batch_size=config.BATCH_NUM)

        # ********** Training  **********
        model.train(train_generator, val_generator,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='all')
