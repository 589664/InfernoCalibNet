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
        project_name="InfernoCalibNetBinary",
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

        wandb.init(
            project=project_name,
            name=f"binary_classification_P7",
            config={
                "model": "ResNet-34",
                "image_size": "256x256",
                "batch_size": train_loader.batch_size,
                "epochs": 17,
                "optimizer": "Adam",
                # "learning_rate": optimizer.param_groups[0]["lr"],
                "learning_rate_backbone": optimizer.param_groups[0]["lr"],
                "learning_rate_classifier": optimizer.param_groups[1]["lr"],
                "weight_decay": optimizer.param_groups[0]["weight_decay"],
                "lr_step_size": scheduler.step_size if scheduler else None,
                "lr_gamma": scheduler.gamma if scheduler else None,
                "architecture": "512 → 128 → 1",
                "dropout": 0.6,
                "loss_fn": type(self.criterion).__name__,
                "device": device.type,
                "pretrained": "IMAGENET1K_V1"
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
            images = images.to(self.device, non_blocking=True)
            labels = labels.float().to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images).squeeze(1)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="binary")

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
                images = images.to(self.device, non_blocking=True)
                labels = labels.float().to(self.device, non_blocking=True)

                outputs = self.model(images).squeeze(1)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).int()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = running_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="binary")

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

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_train_loss = train_loss
                self.early_stop_counter = 0
                model_path = OUT_DIR / "InfernoCalibNetBinary.pth"
                torch.save(self.model.state_dict(), model_path)
                console.print("[bold green]Model saved as InfernoCalibNetBinary.pth[/]")
                wandb.save(str(model_path), base_path=str(OUT_DIR))
            elif train_loss < self.best_train_loss * 0.9:
                self.early_stop_counter += 1
                console.print(
                    f"[bold red]Potential overfitting detected. Early stopping counter: {self.early_stop_counter}/{self.patience}[/]"
                )
            else:
                self.early_stop_counter = 0

            if self.early_stop_counter >= self.patience:
                console.print("[bold red]Early stopping triggered due to overfitting![/]")
                break

        wandb.finish()
