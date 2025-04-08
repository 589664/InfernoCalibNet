# from src.Trainer import Trainer
# from src.Model import InfernoCalibNet
# from src.Dataset import ChestXRayDataset

# from config import OUT_DIR

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader

# from torch.utils.tensorboard import SummaryWriter

# def runTraining():
#     torch.cuda.empty_cache()

#     train_dt = ChestXRayDataset(OUT_DIR / "ml_train.csv", transform=True)
#     val_dt = ChestXRayDataset(OUT_DIR / "ml_val.csv", transform=False)

#     train_loader = DataLoader(
#         train_dt, batch_size=32, shuffle=True, num_workers=8, pin_memory=True
#     )
#     val_loader = DataLoader(
#         val_dt, batch_size=32, shuffle=False, num_workers=8, pin_memory=True
#     )

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = InfernoCalibNet(num_classes=2).to(device)

#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.Adam([
#         {"params": model.base_model.parameters(), "lr": 2e-4},
#         {"params": model.classifier.parameters(), "lr": 1e-3},
#     ], weight_decay=3e-4)

#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.6)

#     trainer = Trainer(
#         model, train_loader, val_loader, criterion, optimizer, device, scheduler
#     )

#     writer = SummaryWriter(log_dir = OUT_DIR / "tensorflow")
#     dummy_input = torch.randn(32, 1, 256, 256).to(device)
#     writer.add_graph(model, dummy_input)

#     trainer.train(num_epochs=10)

# runTraining()


from src.Trainer import Trainer
from src.Model import InfernoCalibNet
from src.Dataset import ChestXRayDataset

from config import OUT_DIR

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from itertools import product

def run_batch_training():
    torch.cuda.empty_cache()

    train_dt = ChestXRayDataset(OUT_DIR / "ml_train.csv", transform=True)
    val_dt = ChestXRayDataset(OUT_DIR / "ml_val.csv", transform=False)

    train_loader = DataLoader(
        train_dt, batch_size=32, shuffle=True, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dt, batch_size=32, shuffle=False, num_workers=8, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_lrs = [1e-4, 2e-4]
    classifier_lrs = [5e-4, 1e-3]
    weight_decays = [1e-4, 3e-4]
    gammas = [0.5, 0.6]
    step_sizes = [3, 5]
    num_epochs_options = [10, 15]

    combinations = list(product(base_lrs, classifier_lrs, weight_decays, gammas, step_sizes))

    for run_id, ((base_lr, clf_lr, wd, gamma, step_size), num_epochs) in enumerate(product(combinations, num_epochs_options)):
        model = InfernoCalibNet(num_classes=2).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam([
            {"params": model.base_model.parameters(), "lr": base_lr},
            {"params": model.classifier.parameters(), "lr": clf_lr},
        ], weight_decay=wd)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        run_name = f"run_{run_id}_base{base_lr}_clf{clf_lr}_wd{wd}_gamma{gamma}_step{step_size}_epochs{num_epochs}"
        writer = SummaryWriter(log_dir=OUT_DIR / "tensorflow" / run_name)

        dummy_input = torch.randn(32, 1, 256, 256).to(device)
        writer.add_graph(model, dummy_input)

        trainer = Trainer(
            model, train_loader, val_loader, criterion, optimizer, device, scheduler
        )
        trainer.train(num_epochs=num_epochs)

run_batch_training()