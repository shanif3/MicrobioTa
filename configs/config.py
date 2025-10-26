# configs/config.py
import torch


class Config:
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 32
    EMBEDDING_DIM = 8 #8
    NHEAD_EMBEDDING = 4
    NHEAD_CLASSIFIER = 4
    NUM_LAYERS_EMBEDDING = 2
    NUM_LAYERS_ENCODER = 2
    LR_EMBEDDING = 0.001
    LR_CLASSIFIER = 0.001
    WEIGHT_DECAY_EMBEDDING = 0.01
    WEIGHT_DECAY_CLASSIFIER = 0.01
    MAX_EPOCHS_EMBEDDING = 2000
    MAX_EPOCHS_CLASSIFIER = 1500
    REG_MODE = 'exponent_on_h'
    REG_LAMBDA = 0.8
    LABEL_SMOOTHING = 0.1
    DATA_DIR = './'
    CHECKPOINT_DIR = './checkpoints'
    TARGET_COL_NAME = ['Tag']
    FREEZE_EMBEDDING_CLASSIFIER = False
    AGGREGATOR_DROPOUT_CLASSIFIER = 0.5
    DROPOUT_LINEAR_CLASSIFIER = 0.3
    DROPOUT_INPUT_CLASSIFIER = 0.3
    DROPOUT_TRANSFORMER= 0.3
    HIDDEN_DIM_TRANSFORMER_EMBEDDING= 256
    HIDDEN_DIM_CLASSIFIER1= 64 #32
    HIDDEN_DIM_CLASSIFIER2= 32 #16

    WANDB_NAME_EMBEDDING = 'EMBEDDING'
    WANDB_NAME_CLASSIFIER = 'CLASSIFIER'
    EARLY_STOPPING_PATIENCE_EMBEDDING = 30
    EARLY_STOPPING_PATIENCE_CLASSIFIER = 40 #10
    FLAG_TYPE_RUN='enter_type'
    RANDOM_INITIALIZE_EMBEDDINGS = False
    path_of_dataset_to_lodo=None

    MODEL_TYPE='score_model1'
    CONTRASTIVE_LAMBDA = 0.1

    CONTRASTIVE_MARGIN = 1.0
    LR_FINE_TUNE_W=0.01

    NUM_LAYERS_CLASSIFIER=2

    @classmethod
    def to_dict(cls):
        return {attr: getattr(cls, attr) for attr in dir(cls) if not attr.startswith("__") and not callable(getattr(cls, attr))}
