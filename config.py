import os
from pathlib import Path

# Normalization values
MEAN = 139.45 / 255.0
STD = 61.93 / 255.0

# Param optimizer variables
DATASET_SIZE = 110000
GPU_MEMORY = 16
INPUT_IMAGE_SIZE = (300, 300)

# Model and Data Parameters
IMG_SIZE = 300
BATCH_SIZE = 20
EPOCHS = 50
LEARNING_RATE = 0.0001
NUM_CLASSES = 15
NUM_WORKERS = 4
DROPOUT_RATE = 0.3

TRAIN_SIZE = 7000
TEST_SIZE = 3000
VAL_SIZE = 1000

# Define paths relative to the root directory
ROOT_DIR = Path(__file__).parent

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "data" / "models"

# Directory for wandb files
os.environ["WANDB_DIR"] = str(ROOT_DIR / "data")
