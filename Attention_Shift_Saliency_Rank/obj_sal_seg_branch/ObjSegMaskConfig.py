from obj_sal_seg_branch.MaskRCNNConfig import Config


# Config for Object Saliency Segmentation Mask
class ObjSegMaskConfig(Config):
    """Configuration for training on the Visual Genome Dataset.
        Derives from the base Config class and overrides some values.
        """
    # Give the configuration a recognizable nameNUM_CLASSES
    NAME = "Obj_Sal_Seg_Mask_Config"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + Salient Object

    MAX_GT_INSTANCES = 5

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50
    # STEPS_PER_EPOCH = 98077
    # VALIDATION_STEPS = 5000

    # # We use a GPU with 12GB memory, which can fit two images.
    # # Adjust down if you use a smaller GPU.
    # IMAGES_PER_GPU = 1
    # # Skip detections with < 70% confidence
    # DETECTION_MIN_CONFIDENCE = 0.7
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    USE_MINI_MASK = False

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7
    # DETECTION_MIN_CONFIDENCE = 0

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    TRAIN_ROIS_PER_IMAGE = 64
    # TRAIN_ROIS_PER_IMAGE = 32

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "obj_sal_seg_class_loss": 1.,
        "obj_sal_seg_bbox_loss": 1.,
        "obj_sal_seg_mask_loss": 1.
    }

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small

    W_REGULARIZER = 0.005
    DROPOUT_VAL = 0.5

    # Network
    NET_IMAGE_SIZE = 1024

    # Reduce from ROIAlign 1024 ->
    OBJ_FEAT_SIZE = 64
