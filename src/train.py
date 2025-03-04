from src.Dataset import ChestXRayDataset
from src.CNNModel import ResNetBinaryClassifier
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
    model = ResNetBinaryClassifier().to(device)

    # ** Freeze Early Layers (Low-Level Features) **
    # for param in model.backbone.conv1.parameters():
    #     param.requires_grad = False
    # for param in model.backbone.layer1.parameters():
    #     param.requires_grad = False
    # for param in model.backbone.layer2.parameters():
    #     param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=2, min_lr=1e-6
    )

    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, device, scheduler
    )
    trainer.train(num_epochs=10)


runTraining()
