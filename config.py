import os
from pathlib import Path

# Normalization values
MEAN = (0.5463, 0.5463, 0.5463)
STD = (0.2366, 0.2366, 0.2366)

# Model and Data Parameters
IMG_SIZE = 224
EPOCHS = 64
BATCH_SZ = 24
LR = 0.0001
NUM_CL = 14
NUM_WRKRS = 8
DROP_RATE = 0.3
PATIENCE = 8

# Dataset distribution
RAND_STATE = 42
STRATIFY_COL = "Finding Labels"
TRAIN_PCT = 0.8
VAL_PCT = 0.10
TEST_PCT = 0.05

DISEASE_LABELS = [
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
CSV_PATH = ROOT_DIR / "data" / "raw" / "xraysMD.csv"
OUT_DIR = ROOT_DIR / "data" / "refined"
MODEL_DIR = ROOT_DIR / "data" / "models" / "bestModel.pth"

# Directory for wandb files
os.environ["WANDB_DIR"] = str(ROOT_DIR / "data")
