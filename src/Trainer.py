import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
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
            outputs = self.model(images).squeeze()
            loss = self.criterion(outputs, labels.float())

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())  # Store raw logits
            all_labels.extend(labels.cpu().numpy())

        # Convert logits to probabilities
        prob_preds = torch.sigmoid(torch.tensor(all_preds)).numpy()
        binary_preds = (prob_preds > 0.5).astype(int)

        avg_loss = running_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, binary_preds)
        roc_auc = roc_auc_score(all_labels, prob_preds)
        f1 = f1_score(all_labels, binary_preds)

        # Log metrics in Weights & Biases
        wandb.log(
            {
                "train_loss": avg_loss,
                "train_accuracy": accuracy,
                "train_roc_auc": roc_auc,
                "train_f1": f1,
            },
            step=self.epoch,
        )

        return avg_loss, accuracy, roc_auc, f1

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

                outputs = self.model(images).squeeze()
                loss = self.criterion(outputs, labels.float())

                running_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())  # Store raw logits
                all_labels.extend(labels.cpu().numpy())

        # Convert logits to probabilities
        prob_preds = torch.sigmoid(torch.tensor(all_preds)).numpy()
        binary_preds = (prob_preds > 0.5).astype(int)

        avg_loss = running_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, binary_preds)
        roc_auc = roc_auc_score(all_labels, prob_preds)
        f1 = f1_score(all_labels, binary_preds)

        # Log validation metrics in Weights & Biases
        wandb.log(
            {
                "val_loss": avg_loss,
                "val_accuracy": accuracy,
                "val_roc_auc": roc_auc,
                "val_f1": f1,
            },
            step=self.epoch,
        )

        return avg_loss, accuracy, roc_auc, f1

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.epoch = epoch
            console.print(f"\n[bold yellow]Epoch {epoch+1}/{num_epochs}[/]")
            train_loss, train_acc, train_auc, train_f1 = self.train_one_epoch()
            val_loss, val_acc, val_auc, val_f1 = self.validate()

            if self.scheduler:
                self.scheduler.step(val_loss)

            console.print(
                f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} | AUC: {train_auc:.4f} | F1: {train_f1:.4f}"
            )
            console.print(
                f"Val Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | AUC: {val_auc:.4f} | F1: {val_f1:.4f}\n"
            )

        wandb.finish()


# Example Usage
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ResNetModel(num_classes=1)
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
# trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, scheduler)
# trainer.train(num_epochs=10)
