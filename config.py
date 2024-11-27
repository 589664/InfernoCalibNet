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
BATCH_SIZE = 24
EPOCHS = 64
LEARNING_RATE = 0.0001
NUM_CLASSES = 15
NUM_WORKERS = 8
DROPOUT_RATE = 0.3

TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

DROPPUT_PATIENCE = 10

# Target ratios for different conditions (in percentage as decimal)
DISEASE_CLASSES = {
    "Atelectasis": 0.08,
    "Cardiomegaly": 0.05,
    "Consolidation": 0.05,
    "Edema": 0.03,
    "Effusion": 0.08,
    "Emphysema": 0.04,
    "Fibrosis": 0.04,
    "Hernia": 0.01,
    "Infiltration": 0.10,
    "Mass": 0.06,
    "No Finding": 0.15,
    "Nodule": 0.05,
    "Pleural_Thickening": 0.04,
    "Pneumonia": 0.05,
    "Pneumothorax": 0.06,
}


# Define paths relative to the root directory
ROOT_DIR = Path(__file__).parent

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "data" / "models"

# Directory for wandb files
os.environ["WANDB_DIR"] = str(ROOT_DIR / "data")
