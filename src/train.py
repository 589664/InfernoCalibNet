from src.Trainer import Trainer
from src.Model import InfernoCalibNet
from src.Dataset import ChestXRayDataset

from config import (
    OUT_DIR,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def runTraining():
    torch.cuda.empty_cache()

    train_dt = ChestXRayDataset(OUT_DIR / "binary_train.csv", transform=True)
    val_dt = ChestXRayDataset(OUT_DIR / "binary_val.csv", transform=False)

    train_loader = DataLoader(
        train_dt, batch_size=16, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dt, batch_size=16, shuffle=False, num_workers=8, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InfernoCalibNet(num_classes=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam([
        {"params": model.base_model.parameters(), "lr": 5e-5},
        {"params": model.classifier.parameters(), "lr": 2e-4},
    ], weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.6)

    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, device, scheduler
    )
    trainer.train(num_epochs=17)

runTraining()
