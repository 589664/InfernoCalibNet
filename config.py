import os
from pathlib import Path

# Normalization values
# MEAN = 0.5463
# STD = 0.2366
MEAN = 0.485
STD = 0.229
CHANNELS = 1

# Model and Data Parameters
IMG_SIZE = 224
EPOCHS = 25
BATCH_SZ = 32
LR = 0.0004
NUM_CL = 13
NUM_WRKRS = 8
DROP_RATE = 0.46
PATIENCE = 7

# Dataset distribution
RAND_STATE = 42
TRAIN_PCT = 0.75
VAL_PCT = 0.15
TEST_PCT = 0.10

DISEASE_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]

WANDB = "InfernoCalibNet_project"

# Define paths relative to the root directory
ROOT_DIR = Path(__file__).parent

XRAY_DIR = ROOT_DIR / "data" / "raw" / "xrays"
CSV_PATH = ROOT_DIR / "data" / "raw" / "xraysMD.csv"
OUT_DIR = ROOT_DIR / "data" / "refined"
MODEL_DIR = ROOT_DIR / "data" / "models" / "bestModel.pth"
OPTUNA_PATH = ROOT_DIR / "data" / "refined" / "hyperParamXray_trial_1.db"

# Directory for wandb files
os.environ["WANDB_DIR"] = str(ROOT_DIR / "data")
