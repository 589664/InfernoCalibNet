import config
from InquirerPy import inquirer
from rich.console import Console

# torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

# custom
from src.ModelInspector import ModelInspector
from src.ICNTrainer import ICNTrainer
from src.XRayDataset import XRayDataset
from src.utils import compute_class_weights, print_label_statistics
from src.prePro import preprocess_metadata, split_data

# Using constants from config
mean = config.MEAN
std = config.STD
img_size = config.IMG_SIZE
batch_size = config.BATCH_SIZE
num_classes = config.NUM_CLASSES
learning_rate = config.LEARNING_RATE
epochs = config.EPOCHS
num_wrks = config.NUM_WORKERS
dropout_rate = config.DROPOUT_RATE

# Paths to directories
raw_dir = config.RAW_DIR
processed_dir = config.PROCESSED_DIR
model_dir = config.MODEL_DIR

train_size = config.TRAIN_SIZE
val_size = config.VAL_SIZE
test_size = config.TEST_SIZE
disease_classes = config.DISEASE_CLASSES

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
                raw_dir,
                processed_dir / "xraysMD.csv",
            )

            # Split the data into train, validation, and test sets
            train_df, val_df, test_df = split_data(
                filtered_df,
                train_ratio=train_size,
                val_ratio=val_size,
                test_ratio=test_size,
                no_finding_ratio=0.10,
            )

            # Print label statistics for each set
            all_labels = list(
                set([label for labels in filtered_df["Labels"] for label in labels])
            )
            print("\Initial Set Label Statistics:")
            print_label_statistics(filtered_df, all_labels)
            print("\nTraining Set Label Statistics:")
            print_label_statistics(train_df, all_labels)
            print("\nValidation Set Label Statistics:")
            print_label_statistics(val_df, all_labels)
            print("\nTest Set Label Statistics:")
            print_label_statistics(test_df, all_labels)

            train_dataset = XRayDataset(
                dataframe=train_df,
                image_dir=raw_dir / "xrays",
                img_size=img_size,
                mean=mean,
                std=std,
            )

            val_dataset = XRayDataset(
                dataframe=val_df,
                image_dir=raw_dir / "xrays",
                img_size=img_size,
                mean=mean,
                std=std,
            )

            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_wrks
            )

            self.val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_wrks
            )

            class_weights, weights_by_label, label_counts = compute_class_weights(
                train_df, disease_classes
            )

            # Print results in a cleaner format
            print("Class Weights:")
            for label, weight in weights_by_label.items():
                print(f"  {label}: {weight:.4f}")

            # Print label counts in a table-like format
            print("\nLabel Counts:")
            print(f"{'Label':<20} {'Positive Count':<15} {'Negative Count':<15}")
            print("-" * 50)
            for label, counts in label_counts.items():
                print(
                    f"{label:<20} {counts['positive_count']:<15} {counts['negative_count']:<15}"
                )

            # Save DataFrame to a CSV file in a given location
            output_csv_path = "data/raw/train.csv"
            train_df.to_csv(output_csv_path, index=False)
            print(f"\nDataFrame saved to {output_csv_path}")

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.cuda.empty_cache()

            # Convert class weights to PyTorch tensor and move it to the device
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(
                self.device
            )
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor)

            # Load and modify the EfficientNet B3 model
            self.model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)

            self.model.features[0][0] = nn.Conv2d(
                1,
                self.model.features[0][0].out_channels,
                kernel_size=self.model.features[0][0].kernel_size,
                stride=self.model.features[0][0].stride,
                padding=self.model.features[0][0].padding,
                bias=False,
            )

            self.model.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),  # Dropout layer for regularization
                nn.Linear(
                    self.model.classifier[1].in_features, num_classes
                ),  # Output layer for final predictions
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

    def inspect_model(self):
        config = {
            "model": self.model,
            "model_path": model_dir / "best_model.pth",
            "input_size": img_size,
            "mean": mean,
            "std": std,
        }
        inspector = ModelInspector(config)
        weights, biases = inspector.get_class_weights_and_biases()
        # print("Class Weights:", weights)
        # print("Class Biases:", biases)
        # predictions = inspector.predict(raw_dir / "xrays" / "00000008_001.png")
        # print("Predictions:", predictions)


def main():
    pipeline_manager = PipelineManager()

    # Define options to call methods on the pipeline manager
    options = {
        "Preprocess Data": pipeline_manager.preprocess_data,
        "Train Model": pipeline_manager.train_model,
        "Inspect Model": pipeline_manager.inspect_model,
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
