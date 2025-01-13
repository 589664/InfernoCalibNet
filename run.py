from config import LR, BATCH_SZ, NUM_WRKRS, IMG_SIZE, CHANNELS

import optuna
import torch
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader

from torchinfo import summary
from InquirerPy import inquirer
from rich import print

from src.NNModels import XrayResNet
from src.ICNTrainer import ICNTrainer
from src.XRayDataset import XrayDataset


def optimize_hyperparameters() -> None:
    def objective(trial: optuna.trial.Trial) -> float:
        # Suggest hyperparameters
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        patience = trial.suggest_int("patience", 3, 10)
        optimizer_type = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
        scheduler_type = trial.suggest_categorical(
            "scheduler", ["ReduceLROnPlateau", "StepLR"]
        )

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
        model = XrayResNet()

        # Define criterion
        criterion = BCEWithLogitsLoss()

        # Define optimizer
        if optimizer_type == "Adam":
            optimizer = Adam(model.parameters(), lr=lr)
        elif optimizer_type == "SGD":
            optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)

        # Define scheduler
        if scheduler_type == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", patience=patience, factor=0.1
            )
        elif scheduler_type == "StepLR":
            scheduler = StepLR(optimizer, step_size=3, gamma=0.8)

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
        storage="sqlite:///hyperparamXray.db",  # Save to SQLite database
        study_name="hyperparamXray_study",
    )
    study.optimize(objective, n_trials=50)

    # Print best trial
    print("Best trial:", study.best_trial)


def main() -> None:
    # Load datasets
    train_dataset = XrayDataset(split="train", augmentations=True)
    validate_dataset = XrayDataset(split="validate", augmentations=False)
    test_dataset = XrayDataset(split="test", augmentations=False)

    # Access and print the first row of the dataset
    image, label = test_dataset[0]
    print("Image Tensor Shape:", image.shape)
    print("Label Tensor:", label)

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SZ,
        shuffle=False,
        num_workers=NUM_WRKRS,
        pin_memory=True,
    )

    # Print one batch for verification
    for images, labels in train_loader:
        print("Batch Images Shape:", images.shape)
        print("Batch Labels Shape:", labels.shape)
        break

    # Initialize model and print summary
    model = XrayResNet()
    summary(model, input_size=(BATCH_SZ, CHANNELS, IMG_SIZE, IMG_SIZE), depth=2)

    # Define training components
    criterion = BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    loaders = {"train": train_loader, "val": val_loader}

    # Initialize and run trainer
    trainer = ICNTrainer(
        device=torch.device("cuda"),
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    trainer.fit()


if __name__ == "__main__":
    mode = inquirer.select(
        message="Choose mode:",
        choices=["train", "optimize", "exit"],
    ).execute()

    if mode == "optimize":
        optimize_hyperparameters()
    elif mode == "train":
        main()
    elif mode == "exit":
        print("[yellow]Exiting.")
    else:
        print("[red]Invalid mode selected.")
