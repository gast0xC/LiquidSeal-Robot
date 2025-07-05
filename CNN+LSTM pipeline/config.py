# config_module.py

import os
from typing import Dict, Tuple

# ======================
#  PATH CONFIGURATIONS
# ======================

BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER: str = os.path.join(BASE_DIR, "dataset")

# ======================
#  IMAGE CONFIGURATIONS
# ======================

IMAGE_HEIGHT: int = 224
IMAGE_WIDTH: int = 224
IMAGE_CHANNELS: int = 3
IMAGE_SIZE: Tuple[int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH)

# ======================
#  COMMAND CONFIGURATIONS
# ======================

OUTPUT_TIME_STEPS: int = 100
COMMAND_TYPE_MAPPING: Dict[str, int] = {
    "CP_S": 0,  # Command Start
    "CP_P": 1,  # Command Point
    "ARC": 2,   # Arc
    "CP_E": 3   # Command End
}

# ======================
#  ENVIRONMENT SETTINGS
# ======================

DEBUG: bool = True  # Set to True for development mode, False for production mode

# ======================
#  TRAINING CONFIGURATIONS
# ======================

if DEBUG:
    BATCH_SIZE: int = 32
    EPOCHS: int = 20
    LEARNING_RATE: float = 0.001  # Slower learning rate for debugging
    print(
        "Running in DEVELOPMENT mode: "
        f"EPOCHS set to {EPOCHS}, BATCH_SIZE set to {BATCH_SIZE}, LEARNING_RATE reduced to {LEARNING_RATE}."
    )
else:
    BATCH_SIZE: int = 32
    EPOCHS: int = 20
    LEARNING_RATE: float = 0.004
    print("Running in PRODUCTION mode.")

# ======================
#  MODEL CONFIGURATIONS
# ======================

LSTM_UNITS: int = 64       # Number of units in the LSTM layers
DENSE_UNITS: int = 128     # Dense layer size
DROPOUT_RATE: float = 0.3  # Dropout rate for regularization



# ENV = os.getenv("ENV", "production")  # 'development' or 'production'
# set ENV=development
#DEBUG = ENV == "development"
#if DEBUG:
#    EPOCHS = 5  # Shorter training for debugging
#    BATCH_SIZE = 8
#    print("Running in DEVELOPMENT mode: EPOCHS set to 5, BATCH_SIZE set to 8")
#else:
#    print("Running in PRODUCTION mode")
