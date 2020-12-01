from conf.BasicModelConfig import BasicModelConfig
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.metrics import AUC, Accuracy, CategoricalAccuracy, BinaryCrossentropy
from tensorflow.python.keras.optimizers import Adam
from conf.BasicModelConfig import BasicModelConfig

class DINModelConfig(BasicModelConfig):
    # Common
    DATA_PATH = '/Users/liangxiao/DeepCTRModel/dataset/amazon/remap.pkl'

    # Network
    ATTENTION_HIDDEN_UNITS = [80, 40]
    FFN_HIDDEN_UNITS = [512, 256, 256]

    ATTENTION_ACTIVATION = 'prelu'
    FFN_ACTIVATION = 'prelu'

    EMBEDDING_DIM = 8

    DROUPOUT_OR_NOT = True

    DROUPOUT_RATIO = 0.5

    ACTIVATION = 'relu'

    MAXLEN = 40

    # TRAIN & TEST
    BATCH_SIZE = 8192

    VALIDATION_RATIO = 0.1

    TEST_RATIO = 0.1

    LEARNING_RATE = 0.001

    OPTIMIZER = 'adam'

    LOSS_FUNCTION = binary_crossentropy

    EARLY_STOPPING_OR_NOT = True

    METRIC = [AUC(), BinaryCrossentropy()]

    EPOCHS = 10