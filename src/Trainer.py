import wandb
import torch
import warnings
import numpy as np
from tqdm.rich import tqdm
from rich.console import Console
from tqdm import TqdmExperimentalWarning
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
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
        project_name="InfernoCalibNetMultilabel",
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

        if not hasattr(self, "logit_stats_table"):
            self.logit_stats_table = wandb.Table(columns=["label", "value", "epoch"])

        wandb.init(
            project=project_name,
            sync_tensorboard=True,
            name=f"ML_Effusion_Atelectasis",
            config={
                "model": "ResNet-34",
                "image_size": "256x256",
                "batch_size": train_loader.batch_size,
                "epochs": 10,
                "optimizer": "Adam",
                "learning_rate_backbone": optimizer.param_groups[0]["lr"],
                "learning_rate_classifier": optimizer.param_groups[1]["lr"],
                "weight_decay": optimizer.param_groups[0]["weight_decay"],
                "lr_step_size": scheduler.step_size if scheduler else None,
                "lr_gamma": scheduler.gamma if scheduler else None,
                "architecture": "512 → 128 → 2",
                "dropout": 0.6,
                "loss_fn": type(self.criterion).__name__,
                "device": device.type,
                "pretrained": "IMAGENET1K_V1",
                "notes": "PA/AP"
            },
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        all_probs, all_labels = [], []
        all_outputs = []

        console.print(f"[bold blue]Training Epoch {self.epoch + 1}...")
        for batch in tqdm(
            self.train_loader,
            desc=f"[green]Epoch {self.epoch + 1}: Training",
            leave=False,
        ):
            images, labels = batch
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            all_outputs.append(outputs.detach().cpu().numpy())

            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            labels_np = labels.cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels_np)

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds_thresh = (all_probs > 0.5).astype(int)

        avg_loss = running_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, preds_thresh)
        f1 = f1_score(all_labels, preds_thresh, average="macro")

        try:
            auroc_macro = roc_auc_score(all_labels, all_probs, average="macro")
            auroc_per_class = roc_auc_score(all_labels, all_probs, average=None)
        except ValueError:
            auroc_macro = float("nan")
            auroc_per_class = [float("nan")] * all_labels.shape[1]

        outputs_np = np.concatenate(all_outputs, axis=0)
        logit_mean = outputs_np.mean(axis=0)
        logit_std = outputs_np.std(axis=0)
        logit_min = outputs_np.min(axis=0)
        logit_max = outputs_np.max(axis=0)

        wandb.log({
            "epoch": self.epoch,
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "train_f1": f1,
            "train_auroc_macro": auroc_macro,
            "train_auroc_effusion": auroc_per_class[0],
            "train_auroc_atelectasis": auroc_per_class[1],
            "Logit Stats/Effusion Mean": logit_mean[0],
            "Logit Stats/Effusion Std": logit_std[0],
            "Logit Stats/Effusion Min": logit_min[0],
            "Logit Stats/Effusion Max": logit_max[0],
            "Logit Stats/Atelectasis Mean": logit_mean[1],
            "Logit Stats/Atelectasis Std": logit_std[1],
            "Logit Stats/Atelectasis Min": logit_min[1],
            "Logit Stats/Atelectasis Max": logit_max[1],
        })

        return avg_loss, accuracy, f1, auroc_macro

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        all_probs, all_labels = [], []

        with torch.no_grad():
            console.print(f"[bold green]Validating Epoch {self.epoch + 1}...")
            for batch in tqdm(
                self.val_loader,
                desc=f"[blue]Epoch {self.epoch + 1}: Validating",
                leave=False,
            ):
                images, labels = batch
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().numpy()
                labels_np = labels.cpu().numpy()

                all_probs.extend(probs)
                all_labels.extend(labels_np)

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds_thresh = (all_probs > 0.5).astype(int)

        avg_loss = running_loss / len(self.val_loader)
        f1 = f1_score(all_labels, preds_thresh, average="macro")
        accuracy = accuracy_score(all_labels, preds_thresh)

        try:
            auroc_macro = roc_auc_score(all_labels, all_probs, average="macro")
            auroc_per_class = roc_auc_score(all_labels, all_probs, average=None)
        except ValueError:
            auroc_macro = float("nan")
            auroc_per_class = [float("nan")] * all_labels.shape[1]

        # prepare data for wandb multilabel-compatible logging (flattened approach)
        y_true_flat = []
        y_pred_flat = []
        for i in range(len(all_labels)):
            for j in range(all_labels.shape[1]):
                if all_labels[i][j] == 1:
                    y_true_flat.append(j)
                    y_pred_flat.append(all_probs[i])


        wandb.log({
            "epoch": self.epoch,
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
            "val_f1": f1,
            "val_auroc_macro": auroc_macro,
            "val_auroc_effusion": auroc_per_class[0],
            "val_auroc_atelectasis": auroc_per_class[1],
            "val_roc_curve": wandb.plot.roc_curve(y_true_flat, y_pred_flat),
            "val_pr_curve": wandb.plot.pr_curve(y_true_flat, y_pred_flat),
        })

        return avg_loss, accuracy, f1, auroc_macro

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.epoch = epoch
            console.print(f"\n[bold yellow]Epoch {epoch+1}/{num_epochs}[/]")
            train_loss, train_acc, train_f1, train_auc = self.train_one_epoch()
            val_loss, val_acc, val_f1, val_auc = self.validate()

            wandb.log({
                "epoch": self.epoch,
                "loss_gap": train_loss - val_loss,
            })

            if self.scheduler:
                self.scheduler.step()

            console.print(
                f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f} | F1: {train_f1:.4f} | AUROC: {train_auc:.4f}"
            )
            console.print(
                f"Val Loss:   {val_loss:.4f} | Accuracy: {val_acc:.4f} | F1: {val_f1:.4f} | AUROC: {val_auc:.4f}\n"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_train_loss = train_loss
                self.early_stop_counter = 0
                model_path = OUT_DIR / "InfernoCalibNetML.pth"
                torch.save(self.model.state_dict(), model_path)
                console.print("[bold green]Model saved as InfernoCalibNetML.pth[/]")
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
