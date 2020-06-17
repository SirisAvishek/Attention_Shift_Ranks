from Config import Config


# Config for Saliency Rank
class RankModelConfig(Config):
    """Configuration for training on the Visual Genome Dataset.
        Derives from the base Config class and overrides some values.
        """
    # Give the configuration a recognizable nameNUM_CLASSES
    NAME = "Rank_Model_Config"

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

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0

    LOSS_WEIGHTS = {
        "rank_loss": 1.0,
    }

    # Use when using RPN - i.e. training segmentation mask branch only with Proposal Layer, RPN ...
    # TRAIN_ROIS_PER_IMAGE = 64
    # TRAIN_ROIS_PER_IMAGE = 32

    # For training whole network with ranking
    TRAIN_ROIS_PER_IMAGE = 10

    # Use this directly when generating GT mask data
    # MASK_OUT_SHAPE = (28, 28)

    SAL_OBJ_NUM = 30

    # Rank Classes
    # 0 = BG, 1 = Rank_5, 2 = Rank_4, 3 = Rank_3, 4 = Rank_2, 5 = Rank_1
    SAL_RANK = 1 + 5

    # Network
    NET_IMAGE_SIZE = 1024

    OBJ_FEAT_SIZE = 1024

    BOTTLE_NECK_SIZE = 512

    NUM_ATTN_HEADS = 4

    BATCH_NUM = 8

    BACKBONE_FEAT_RESIZE = 32

    RANK_FEAT_SIZE = 512

    RANK_CLASS_HIDDEN_LAYER_DIMS = [512, 512, 512]

    # Threshold, convert segmentation mask prediction to binary 0 or 1
    MASK_THRESH = 0.5


class InferenceConfig(RankModelConfig):
    NAME = "Rank_Model_Inference_Config"

    BATCH_NUM = 1
