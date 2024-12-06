import torch
import torch.nn as nn
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score
from tqdm import tqdm
from rich.console import Console
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from config import BATCH_SZ, NUM_WRKRS, LR, EPOCHS, PATIENCE


class ICNTrainer:
    def __init__(
        self,
        model,
        train_ds,
        val_ds,
        batch_size: int = BATCH_SZ,
        learning_rate: float = LR,
    ):
        self.console = Console()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()

        self.model = model().to(self.device)
        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WRKRS
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WRKRS
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.metric_auc = MultilabelAUROC(num_labels=15).to(self.device)
        self.metric_f1 = MultilabelF1Score(num_labels=15).to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=1e-5,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=EPOCHS, eta_min=0)
        self.epochs = EPOCHS
        self.patience = PATIENCE
        self.scaler = GradScaler()

    def train_one_epoch(self):
        self.model.train()
        train_ls_sum = 0.0
        for images, labels in tqdm(self.train_loader, desc="Training"):
            images, labels = images.to(self.device), labels.to(self.device).float()

            self.optimizer.zero_grad()
            with autocast(device_type=self.device.type):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Ensure labels are float32 for metrics
            labels = labels.float()

            self.metric_auc.update(outputs, labels.long())
            self.metric_f1.update(outputs, labels.long())

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_ls_sum += loss.item()

        avg_train_ls = train_ls_sum / len(self.train_loader)
        self.console.log(
            f"Training Loss: {avg_train_ls:.4f}, AUC: {self.metric_auc.compute():.4f}, F1 Score: {self.metric_f1.compute():.4f}"
        )
        return avg_train_ls

    def validate(self):
        self.model.eval()
        valid_ls_sum = 0.0
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validating"):
                images, labels = images.to(self.device), labels.to(self.device).float()

                with autocast(device_type=self.device.type):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                valid_ls_sum += loss.item()

                # Ensure labels are float32 for metrics
                labels = labels.float()

                self.metric_auc.update(outputs, labels.long())
                self.metric_f1.update(outputs, labels.long())

        avg_valid_loss = valid_ls_sum / len(self.val_loader)
        auc_score = self.metric_auc.compute()
        f1_score = self.metric_f1.compute()
        self.console.log(
            f"Validation Loss: {avg_valid_loss:.4f}, AUC: {auc_score:.4f}, F1 Score: {f1_score:.4f}"
        )

        self.metric_auc.reset()
        self.metric_f1.reset()
        return avg_valid_loss

    def fit(self):
        best_valid_ls = float("inf")
        early_stop_cntr = 0

        for epoch in range(self.epochs):
            self.console.log(f"Epoch [{epoch + 1}/{self.epochs}]")
            train_ls = self.train_one_epoch()
            valid_ls = self.validate()

            self.scheduler.step()

            if valid_ls < best_valid_ls:
                best_valid_ls = valid_ls
                early_stop_cntr = 0
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                early_stop_cntr += 1

            if early_stop_cntr >= self.patience:
                self.console.log("Early stopping triggered")
                break


# Example usage
# trainer = ICNTrainer(model, train_ds, val_ds)
# trainer.fit()
