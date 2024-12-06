import os
from pathlib import Path

# Normalization values
MEAN = (0.5463, 0.5463, 0.5463)
STD = (0.2366, 0.2366, 0.2366)

# Model and Data Parameters
BATCH_SIZE = 24
EPOCHS = 64
LEARNING_RATE = 0.0001
NUM_CLASSES = 15
NUM_WORKERS = 8
DROPOUT_RATE = 0.3
IMG_SIZE = 300

TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15

DROPPUT_PATIENCE = 10

# Target ratios for different conditions (in percentage as decimal)
DISEASE_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "No Finding",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]

# Define paths relative to the root directory
ROOT_DIR = Path(__file__).parent

XRAY_DIR = ROOT_DIR / "data" / "raw" / "xrays"
IN_CSV = ROOT_DIR / "data" / "raw" / "xraysMD.csv"
OUT_CSV = ROOT_DIR / "data" / "refined" / "xraysMD.csv"

# Directory for wandb files
os.environ["WANDB_DIR"] = str(ROOT_DIR / "data")
