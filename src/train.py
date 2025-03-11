from src.Dataset import ChestXRayDataset
from src.Model import InfernoCalibNet
from src.Trainer import Trainer

from config import (
    CSV_PATH,
    XRAY_DIR,
    OUT_DIR,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def runTraining():
    torch.cuda.empty_cache()  # Clear unused GPU memory

    train_dt = ChestXRayDataset(OUT_DIR / "train.csv", transform=True)
    val_dt = ChestXRayDataset(OUT_DIR / "val.csv", transform=False)

    train_loader = DataLoader(
        train_dt, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dt, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InfernoCalibNet(num_classes=3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.63)

    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, device, scheduler
    )
    trainer.train(num_epochs=15)


runTraining()
