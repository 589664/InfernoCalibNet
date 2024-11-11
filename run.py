import time
import config
from InquirerPy import inquirer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeRemainingColumn

# torch
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

# custom
from src.ICNTrainer import ICNTrainer
from src.XRayDataset import XRayDataset
from src.utils import compute_class_weights
from src.prePro import preprocess_metadata, distribution_df_split

# Using constants from config
mean = config.MEAN
std = config.STD
img_size = config.IMG_SIZE
batch_size = config.BATCH_SIZE
num_classes = config.NUM_CLASSES
learning_rate = config.LEARNING_RATE
epochs = config.EPOCHS

# Paths to directories
raw_dir = config.RAW_DIR
processed_dir = config.PROCESSED_DIR
model_dir = config.MODEL_DIR
train_size = config.TRAIN_SIZE
test_size = config.TEST_SIZE

# Initialize Rich console
console = Console()


class PipelineManager:
    def __init__(self):
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.optimizer = None
        self.criterion = None
        self.device = None

    def preprocess_data(self):
        console.print("[bold green]Preprocessing data...[/bold green]")
        with console.status("Processing data...", spinner="dots"):
            # Data preprocessing steps
            filtered_df = preprocess_metadata(
                raw_dir / "xraysMD.csv",
                raw_dir / "xrays",
                processed_dir / "xraysMD.csv",
            )
            train_df, test_df = distribution_df_split(
                filtered_df, train_size=train_size, test_size=test_size
            )
            train_dataset = XRayDataset(
                dataframe=train_df,
                image_dir=raw_dir / "xrays",
                img_size=img_size,
                mean=mean,
                std=std,
            )
            val_dataset = XRayDataset(
                dataframe=test_df,
                image_dir=raw_dir / "xrays",
                img_size=img_size,
                mean=mean,
                std=std,
            )
            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
            )
            self.val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=1
            )
            class_weights = compute_class_weights(train_df, num_classes)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.cuda.empty_cache()

            # Convert class weights to PyTorch tensor and move it to the device
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(
                self.device
            )
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)

            # Load and modify the EfficientNet B3 model
            self.model = models.efficientnet_b3(pretrained=True)
            self.model.features[0][0] = nn.Conv2d(
                1,
                self.model.features[0][0].out_channels,
                kernel_size=self.model.features[0][0].kernel_size,
                stride=self.model.features[0][0].stride,
                padding=self.model.features[0][0].padding,
                bias=False,
            )
            self.model.classifier[1] = nn.Linear(
                self.model.classifier[1].in_features, num_classes
            )
            self.model = self.model.to(self.device)

            # Set up the optimizer
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        console.print("[bold green]Data preprocessing complete![/bold green]")

    def train_model(self):
        if not all(
            [self.model, self.train_loader, self.optimizer, self.criterion, self.device]
        ):
            console.print("[bold red]Please run preprocessing first![/bold red]")
            return

        # Instantiate ICNTrainer and begin training
        console.print("[bold blue]Training model with ICNTrainer...[/bold blue]")
        trainer = ICNTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            optimizer=self.optimizer,
            criterion=self.criterion,
            device=self.device,
            project_name="inferno-calib-net",
            config={
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
            },
        )

        trainer.fit(epochs=epochs)
        console.print("[bold blue]Model training complete![/bold blue]")


def main():
    pipeline_manager = PipelineManager()

    # Define options to call methods on the pipeline manager
    options = {
        "Preprocess Data": pipeline_manager.preprocess_data,
        "Train Model": pipeline_manager.train_model,
        "Quit": None,
    }

    # Interactive menu loop
    while True:
        choice = inquirer.select(
            message="Select an action:",
            choices=list(options.keys()),
            default="Preprocess Data",
        ).execute()

        if choice == "Quit":
            console.print("[bold red]Exiting...[/bold red]")
            break
        else:
            options[choice]()


if __name__ == "__main__":
    main()
