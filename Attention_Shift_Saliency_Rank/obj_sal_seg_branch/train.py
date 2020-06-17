import imgaug
from obj_sal_seg_branch.ObjSegMaskConfig import ObjSegMaskConfig
from obj_sal_seg_branch.SOSNet import SOSNet
from obj_sal_seg_branch.Obj_Sal_Seg_Dataset import Obj_Sal_Seg_Dataset

DATASET_ROOT = "D:/Desktop/ASSR/"   # Change to your location


if __name__ == '__main__':
    command = "train"

    config = ObjSegMaskConfig()
    config.display()
    log_path = "logs/"

    model = SOSNet(mode="training", config=config, model_dir=log_path)

    # Start from pre-trained weights
    # Load weights
    model_weights = "../weights/mask_rcnn_coco.h5"  # Make sure this is correct or change to location of weight path

    # Exclude layers - since we change the number of classes
    exclude_layers = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]
    print("Exclude Layers: ", exclude_layers)

    print("Loading weights ", model_weights)
    model.load_weights(model_weights, by_name=True, exclude=exclude_layers)

    # Train/Evaluate Model
    if command == "train":
        print("Start Training...")

        # Train Dataset
        dataset_train = Obj_Sal_Seg_Dataset(DATASET_ROOT, "train")

        # Val Dataset
        dataset_val = Obj_Sal_Seg_Dataset(DATASET_ROOT, "val")

        # ********** Training  **********
        # Image Augmentation
        # Right/Left flip 50% of the time
        augmentation = imgaug.augmenters.Fliplr(0.5)

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads',
                    augmentation=augmentation)

        # Training - Stage 2
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=200,
                    layers='all',
                    augmentation=augmentation)
