import os
import optuna
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from rich import print

from config import LR, BATCH_SZ, NUM_WRKRS, IMG_SIZE, CHANNELS, OPTUNA_PATH, OUT_DIR
from src.NNModels import XrayResNet
from src.ICNTrainer import ICNTrainer
from src.XrayDataset import XrayDataset
from src.utils.DatasetTools import preprocess_and_split_csv, calculate_class_weights


def preprocess_and_display_stats() -> None:
    stats = preprocess_and_split_csv()
    print(stats)


def run_training() -> None:
    device = torch.device("cuda")

    # Load datasets
    train_dataset = XrayDataset(split="train", augmentations=True)
    validate_dataset = XrayDataset(split="validate", augmentations=False)
    test_dataset = XrayDataset(split="test", augmentations=False)

    # Create DataLoader objects
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SZ,
        shuffle=True,
        num_workers=NUM_WRKRS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        validate_dataset,
        batch_size=BATCH_SZ,
        shuffle=False,
        num_workers=NUM_WRKRS,
        pin_memory=True,
    )

    # Initialize model and print summary/tensorboard graph
    model = XrayResNet(model_type="resnet50")
    summary(model, input_size=(BATCH_SZ, CHANNELS, IMG_SIZE, IMG_SIZE), depth=2)

    # writer = SummaryWriter(log_dir=os.path.join(OUT_DIR, "tensorboard_logs"))
    # dummy_input = torch.randn(1, CHANNELS, IMG_SIZE, IMG_SIZE).to(device)
    # writer.add_graph(model, dummy_input)
    # writer.close()

    # Freeze all layers except `layer4` and `fc`
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    # Define training components
    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=7, factor=0.4)

    loaders = {"train": train_loader, "val": val_loader}

    # Initialize and run trainer
    trainer = ICNTrainer(
        device=device,
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    trainer.fit(early_stopping=True, gradual_unfreeze=[(5, "layer3"), (10, "layer2")])


def optimize_hyperparams() -> None:
    def objective(trial: optuna.trial.Trial) -> float:
        # Suggest hyperparameters
        lr = trial.suggest_float("learning_rate", 2e-4, 6e-4, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        # Load datasets
        train_dataset = XrayDataset(split="train", augmentations=True)
        validate_dataset = XrayDataset(split="validate", augmentations=False)

        # Create DataLoader objects
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=NUM_WRKRS,
        )
        val_loader = DataLoader(
            validate_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WRKRS,
        )

        # Initialize model
        model = XrayResNet(model_type="resnet50")

        # Define criterion
        criterion = BCEWithLogitsLoss()

        # Define optimizer
        optimizer = Adam(model.parameters(), lr=lr)

        # Define scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.4)

        loaders = {"train": train_loader, "val": val_loader}

        # Initialize trainer
        trainer = ICNTrainer(
            device=torch.device("cuda"),
            model=model,
            loaders=loaders,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        # Train and validate
        trainer.fit(early_stopping=True)

        # Return validation loss for Optuna
        return trainer.best_val_loss

    # Create Optuna study
    study = optuna.create_study(
        direction="minimize",
        storage=f"sqlite:///{OPTUNA_PATH}",  # Save to OUT_DIR
        study_name="hyperParamXray_study",
        load_if_exists=True,  # Resume if the study already exists
    )
    study.optimize(objective, n_trials=50)

    # Print best trial
    print("Best trial:", study.best_trial)
