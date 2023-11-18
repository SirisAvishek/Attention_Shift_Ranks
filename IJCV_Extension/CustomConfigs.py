from Config import Config


# Config for Saliency Rank
class RankModelConfig(Config):
    """Configuration for training on the Visual Genome Dataset.
        Derives from the base Config class and overrides some values.
        """
    # Give the configuration a recognizable nameNUM_CLASSES
    NAME = "Rank_Config"

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False  # Defaulting to False since batch size is often small

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50

    # Using Adam Optimizer - Default value
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    GPU_COUNT = 1

    IMAGES_PER_GPU = 8

    DETECTION_MIN_CONFIDENCE = 0.7

    LOSS_WEIGHTS = {
        "rpn_class_loss": 1.0,
        "rpn_bbox_loss": 1.0,
        "obj_sal_seg_class_loss": 1.0,
        "obj_sal_seg_bbox_loss": 1.0,
        "obj_sal_seg_mask_loss": 1.0,
        "sal_rank_loss": 1.0,
        "obj_sal_edge_loss": 1.0,
    }

    USE_MINI_MASK = False
    # USE_MINI_MASK = True
    # INCREASE MASK SIZE, SO WE DO NOT LOSE SOME OBJECTS IN GT_MASKS
    # MINI_MASK_SHAPE = (224, 224)  # (height, width) of the mini-mask
    # MINI_MASK_SHAPE = (1024, 1024)  # (height, width) of the mini-mask

    # REDUCE TO 5, SINCE ONLY MAX 5 GT SAL OBJECTS
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 5

    # Maximum number of ground truth instances to use in one image
    DETECTION_MAX_INSTANCES = 100

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256  # Default

    # For training whole network end-to-end with ranking
    TRAIN_ROIS_PER_IMAGE = 200  # Default

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # Use this directly when generating GT mask data
    # MASK_OUT_SHAPE = (28, 28)

    SAL_OBJ_NUM = 200  # Default

    TOP_K_DETECTION = 12    # MAX top # of detected objects - used as proposals for saliency ranking

    # Rank Classes
    # 0 = BG, 1 = Rank_5, 2 = Rank_4, 3 = Rank_3, 4 = Rank_2, 5 = Rank_1
    SAL_RANK = 1 + 5

    # Network
    NET_IMAGE_SIZE = 1024

    OBJ_FEAT_SIZE = 1024

    BOTTLE_NECK_SIZE = 512

    NUM_ATTN_HEADS = 4

    BACKBONE_FEAT_RESIZE = 32

    RANK_FEAT_SIZE = 512

    SAL_CLASS_HIDDEN_LAYER_DIMS = [512, 512, 512]
    RANK_CLASS_HIDDEN_LAYER_DIMS = [256, 128, 64]

    # Threshold, convert segmentation mask prediction to binary 0 or 1
    MASK_THRESH = 0.5

    def __init__(self):
        super().__init__()

        # if self.IMAGES_PER_GPU > 1:
        #     self.TRAIN_BN = True

        self.STEPS_PER_EPOCH = 7646 // self.IMAGES_PER_GPU
        self.VALIDATION_STEPS = 1436 // self.IMAGES_PER_GPU


class InferenceConfig(RankModelConfig):
    NAME = "Rank_Model_Inference_Config"

    TRAIN_BN = False

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    SAL_OBJ_NUM = 100
    DETECTION_MAX_INSTANCES = 100
