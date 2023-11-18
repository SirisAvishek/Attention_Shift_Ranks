from Dataset import Dataset
from CustomConfigs import RankModelConfig
from model.ASSRNet import ASSRNet
import DataGenerator
from model import Model
import imgaug

# TODO: Change Here
# Path to dataset
DATASET_ROOT = "...."  # Change to your location

if __name__ == '__main__':
    # TODO: Change Here
    weight_path = "weights/Base_Edge.h5"    # Change to your location

    command = "train"
    config = RankModelConfig()
    config.display()
    log_path = "logs/"
    mode = "training"

    # ********************

    print("Loading ASSRNet Model")

    # TODO: Comment when fine-tuning
    '''For Pre-training'''
    keras_model = Model.build_saliency_rank_model(config, "training", "pre")

    # TODO: Uncomment for fine-tuning, Comment the above
    '''For Fine-tuning'''
    # keras_model = Model.build_saliency_rank_model(config, "training", "fine")

    # ********************

    model_name = "ASSRNet"
    model = ASSRNet(mode=mode, config=config, model_dir=log_path, keras_model=keras_model, model_name=model_name)

    exclude_layers = ["anchors"]    # Updated the constant Lambda(tf.Variable) --> to a ConstLayer
    # exclude_layers = []

    # Load weights
    print("Loading weights ", weight_path)
    model.load_weights(weight_path, by_name=True, exclude=exclude_layers)

    # Train/Evaluate Model
    if command == "train":
        print("Start Training...")

        # ********** Create Datasets
        # Train Dataset
        train_dataset = Dataset(DATASET_ROOT, "train")

        # Val Dataset
        val_dataset = Dataset(DATASET_ROOT, "val")

        # ********** Parameters
        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)
        # augmentation = None

        # ********** Create Data generators
        train_generator = DataGenerator.data_generator(train_dataset, config, shuffle=True,
                                                       augmentation=augmentation,
                                                       batch_size=config.BATCH_SIZE)
        val_generator = DataGenerator.data_generator(val_dataset, config, shuffle=True,
                                                     batch_size=config.BATCH_SIZE)

        # ********** Training  **********
        # TODO: Comment when fine-tuning
        '''For Pre-training'''
        model.train(train_generator, val_generator,
                    learning_rate=config.LEARNING_RATE,
                    epochs=30,
                    layers='heads')

        # TODO: Uncomment for fine-tuning, Comment the above
        '''For Fine-tuning'''
        # model.train(train_generator, val_generator,
        #             learning_rate=config.LEARNING_RATE/1000,
        #             epochs=10,
        #             layers='all',
        #             dlr_split_idx=344)


