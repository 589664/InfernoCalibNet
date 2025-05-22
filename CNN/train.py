import os
import wandb
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from CNN import Trainer, InfernoCalibNet, ChestXRayDataset, OUT_DIR

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def run_batch_training(runs, model_type='resnet50', pretrained=True):
    torch.cuda.empty_cache()
    set_seed(42)

    generator = torch.Generator()
    generator.manual_seed(42)

    train_dt = ChestXRayDataset(OUT_DIR / "ml_train.csv", transform=True)
    val_dt = ChestXRayDataset(OUT_DIR / "ml_val.csv", transform=False)

    train_loader = DataLoader(
        train_dt,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator
    )
    val_loader = DataLoader(
        val_dt,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for run_id, params in enumerate(runs):
        base_lr = params["base_lr"]
        clf_lr = params["clf_lr"]
        wd = params["weight_decay"]
        gamma = params["gamma"]
        step_size = params["step_size"]
        num_epochs = params["num_epochs"]

        wandb.init(
            project="InfernoCalibNetMultilabel",
            sync_tensorboard=True,
            name=f"ML_Effusion_Atelectasis",
            config={
                "model": model_type,
                "pretrained": pretrained,
                "image_size": "256x256",
                "batch_size": train_loader.batch_size,
                "epochs": num_epochs,
                "optimizer": "Adam",
                "learning_rate_backbone": base_lr,
                "learning_rate_classifier": clf_lr,
                "weight_decay": wd,
                "lr_step_size": step_size,
                "lr_gamma": gamma,
                "architecture": "512/2048 → 128 → 2",
                "dropout": 0.6,
                "loss_fn": "BCEWithLogitsLoss",
                "device": device.type,
                "notes": "PA/AP",
            },
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

        model = InfernoCalibNet(
            num_classes=2,
            drop_rate=0.6,
            model_type=model_type,
            pretrained=pretrained,
        ).to(device)

        criterion = nn.BCEWithLogitsLoss()

        optimizer = optim.Adam([
            {"params": model.base_model.parameters(), "lr": base_lr},
            {"params": model.classifier.parameters(), "lr": clf_lr},
        ], weight_decay=wd)

        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )

        writer = SummaryWriter(log_dir=OUT_DIR / "tensorflow")

        dummy_input = torch.randn(32, 1, 256, 256).to(device)
        writer.add_graph(model, dummy_input)

        trainer = Trainer(
            model, train_loader, val_loader, criterion, optimizer, device, scheduler
        )
        trainer.train(num_epochs=num_epochs)

runs = [
    {"base_lr": 1e-4, "clf_lr": 5e-4, "weight_decay": 5e-4, "gamma": 0.6, "step_size": 5, "num_epochs": 15},
]
run_batch_training(runs, model_type='resnet50', pretrained=True)

# Examples:
# runs1 = [
#     {"base_lr": 1e-4, "clf_lr": 5e-4, "weight_decay": 5e-4, "gamma": 0.6, "step_size": 5, "num_epochs": 20},
#     {"base_lr": 1e-4, "clf_lr": 5e-4, "weight_decay": 4e-4, "gamma": 0.6, "step_size": 5, "num_epochs": 20},
#     {"base_lr": 1e-4, "clf_lr": 5e-4, "weight_decay": 5e-4, "gamma": 0.6, "step_size": 5, "num_epochs": 20},
# ]
# run_batch_training(runs, model_type='resnet50', pretrained=False)
# run_batch_training(runs1, model_type='resnet34', pretrained=True)

