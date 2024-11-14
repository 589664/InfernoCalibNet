from pathlib import Path

# Normalization values
MEAN = 139.45 / 255.0
STD = 61.93 / 255.0

# Model and Data Parameters
IMG_SIZE = 300
BATCH_SIZE = 20
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 15
TRAIN_SIZE = 700
TEST_SIZE = 100

# Define paths relative to the root directory
ROOT_DIR = Path(__file__).parent

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "data" / "models"
