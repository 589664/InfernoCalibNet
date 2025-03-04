import os
from pathlib import Path

# Normalization values
MEAN = 0.5463
STD = 0.2366
# MEAN = 0.485
# STD = 0.229
CHANNELS = 1

# Model and Data Parameters
MODEL_TYPE = "resnet50"
IMG_SIZE = 512
EPOCHS = 30
BATCH_SZ = 32
LR = 1e-3
NUM_CL = 1
NUM_WRKRS = 8
DROP_RATE = 0.3
PATIENCE = 5

DISEASE_LABELS = []

# Dataset distribution
RAND_STATE = 2025
TRAIN_PCT = 0.75
VAL_PCT = 0.15
TEST_PCT = 0.10


# Define paths relative to the root directory
ROOT_DIR = Path(__file__).parent

XRAY_DIR = ROOT_DIR / "data" / "raw" / "xrays"
CSV_PATH = ROOT_DIR / "data" / "raw" / "xraysMD.csv"
OUT_DIR = ROOT_DIR / "data" / "refined"
MODEL_DIR = ROOT_DIR / "data" / "models"
OPTUNA_PATH = ROOT_DIR / "data" / "refined" / "hyperParamXray_trial_1.db"

# WANDB
WANDB = "InfernoCalibNet_project"
os.environ["WANDB_DIR"] = str(ROOT_DIR / "data")
