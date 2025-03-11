import torch
from sklearn.metrics import accuracy_score, f1_score
from rich.console import Console
from tqdm.rich import tqdm
import wandb
import warnings
from tqdm import TqdmExperimentalWarning
from config import OUT_DIR

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

console = Console()


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        scheduler=None,
        project_name="InfernoCalibNet",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epoch = 0
        self.patience = 3
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

        # Initialize Weights & Biases
        wandb.init(
            project=project_name,
            config={
                "learning_rate": optimizer.param_groups[0]["lr"],
                "batch_size": train_loader.batch_size,
                "device": device.type,
            },
        )

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []

        console.print(f"[bold blue]Training Epoch {self.epoch + 1}...")
        for batch in tqdm(
            self.train_loader,
            desc=f"[green]Epoch {self.epoch + 1}: Training",
            leave=False,
        ):
            images, labels = batch
            images, labels = images.to(self.device, non_blocking=True), labels.to(
                self.device, non_blocking=True
            )

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        # Log metrics in Weights & Biases
        wandb.log(
            {
                "train_loss": avg_loss,
                "train_accuracy": accuracy,
                "train_f1": f1,
            },
            step=self.epoch,
        )

        return avg_loss, accuracy, f1

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            console.print(f"[bold green]Validating Epoch {self.epoch + 1}...")
            for batch in tqdm(
                self.val_loader,
                desc=f"[blue]Epoch {self.epoch + 1}: Validating",
                leave=False,
            ):
                images, labels = batch
                images, labels = images.to(self.device, non_blocking=True), labels.to(
                    self.device, non_blocking=True
                )

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")

        # Log validation metrics in Weights & Biases
        wandb.log(
            {
                "val_loss": avg_loss,
                "val_accuracy": accuracy,
                "val_f1": f1,
            },
            step=self.epoch,
        )

        return avg_loss, accuracy, f1

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.epoch = epoch
            console.print(f"\n[bold yellow]Epoch {epoch+1}/{num_epochs}[/]")
            train_loss, train_acc, train_f1 = self.train_one_epoch()
            val_loss, val_acc, val_f1 = self.validate()

            if self.scheduler:
                self.scheduler.step()

            console.print(
                f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} | F1: {train_f1:.4f}"
            )
            console.print(
                f"Val Loss:   {val_loss:.4f} | Accuracy: {val_acc:.4f} | F1: {val_f1:.4f}\n"
            )

            # Early stopping logic for overfitting detection
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_train_loss = train_loss
                self.early_stop_counter = 0
                model_path = OUT_DIR / "InfernoCalibNet_model.pth"
                torch.save(self.model.state_dict(), model_path)
                console.print("[bold green]Model saved as InfernoCalibNet_model.pth[/]")
                wandb.save(str(model_path), base_path=str(OUT_DIR))
            elif (
                train_loss < self.best_train_loss * 0.9
            ):  # Check if training loss keeps dropping while val loss stagnates
                self.early_stop_counter += 1
                console.print(
                    f"[bold red]Potential overfitting detected. Early stopping counter: {self.early_stop_counter}/{self.patience}[/]"
                )
            else:
                self.early_stop_counter = 0  # Reset if no overfitting detected

            if self.early_stop_counter >= self.patience:
                console.print(
                    "[bold red]Early stopping triggered due to overfitting![/]"
                )
                break

        wandb.finish()
