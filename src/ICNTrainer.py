import warnings

import torch
from tqdm.rich import tqdm
from tqdm import TqdmExperimentalWarning
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score
from rich.console import Console
import wandb

from config import EPOCHS, PATIENCE, NUM_CL, MODEL_DIR, WANDB, LR, BATCH_SZ

# Suppress TQDM Experimental Warning
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)


class ICNTrainer:
    def __init__(
        self,
        device: torch.device,
        model: torch.nn.Module,
        loaders: dict,
        criterion: torch.nn.Module,
        optimizer: torch.optim,
        scheduler: torch.optim,
        use_wandb: bool = True,
    ) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.console = Console()
        self.use_wandb = use_wandb

        # DataLoaders
        self.train_loader = loaders["train"]
        self.val_loader = loaders["val"]

        # Training Components
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Metrics
        self.auroc = MultilabelAUROC(num_labels=NUM_CL, average="macro").to(self.device)
        self.f1_score = MultilabelF1Score(num_labels=NUM_CL, average="macro").to(
            self.device
        )

        # Learning Rate Warning
        self.best_val_loss = float("inf")
        self.no_improvement_counter = 0

        # Initialize Weights & Biases (WandB) if enabled
        if self.use_wandb:
            wandb.init(
                project=WANDB,
                config={"learning_rate": LR, "batch_size": BATCH_SZ, "epochs": EPOCHS},
            )
            wandb.watch(self.model, log="all", log_freq=100)

        # Freeze all layers except `layer4` and `fc`
        for name, param in self.model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False

    def freeze_layers(self, freeze_until: str) -> None:
        for name, param in self.model.named_parameters():
            param.requires_grad = False  # Freeze all layers
            if freeze_until in name:
                break

    def unfreeze_layers(self, freeze_until: str) -> None:
        for name, param in self.model.named_parameters():
            param.requires_grad = True  # Unfreeze layers
            if freeze_until in name:
                break

    def train(self, epoch: int) -> tuple:
        self.model.train()
        train_loss = 0.0
        self.auroc.reset()
        self.f1_score.reset()

        for batch in tqdm(
            self.train_loader, desc=f"[green]Epoch {epoch + 1}: Training"
        ):
            images, targets = batch
            images, targets = images.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            self.auroc.update(outputs, targets.int())
            self.f1_score.update(outputs, targets.int())

        avg_loss = train_loss / len(self.train_loader)
        train_auroc = self.auroc.compute().item()
        train_f1 = self.f1_score.compute().item()

        # Log metrics to WandB if enabled
        if self.use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "train_auroc": train_auroc,
                    "train_f1": train_f1,
                }
            )

        self.console.log(
            f"[green]Train Loss: {avg_loss:.4f}, Train AUROC: {train_auroc:.4f}, Train F1: {train_f1:.4f}"
        )
        return avg_loss, train_auroc, train_f1

    def validate(self, epoch: int) -> tuple:
        self.model.eval()
        val_loss = 0.0
        self.auroc.reset()
        self.f1_score.reset()

        with torch.no_grad():
            for batch in tqdm(
                self.val_loader, desc=f"[blue]Epoch {epoch + 1}: Validating"
            ):
                images, targets = batch
                images, targets = images.to(self.device), targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

                self.auroc.update(outputs, targets.int())
                self.f1_score.update(outputs, targets.int())

        avg_loss = val_loss / len(self.val_loader)
        val_auroc = self.auroc.compute().item()
        val_f1 = self.f1_score.compute().item()

        # Log metrics to WandB if enabled
        if self.use_wandb:
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "val_loss": avg_loss,
                    "val_auroc": val_auroc,
                    "val_f1": val_f1,
                }
            )

        self.console.log(
            f"[blue]Val Loss: {avg_loss:.4f}, Val AUROC: {val_auroc:.4f}, Val F1: {val_f1:.4f}"
        )
        return avg_loss, val_auroc, val_f1

    def fit(self, early_stopping: bool = True, gradual_unfreeze: list = None) -> None:
        if gradual_unfreeze:
            gradual_unfreeze = sorted(
                gradual_unfreeze, key=lambda x: x[0]
            )  # Sort by epoch
            unfreeze_schedule = iter(gradual_unfreeze)
            next_unfreeze = next(unfreeze_schedule, None)

        for epoch in range(EPOCHS):
            # Gradual unfreezing logic
            if next_unfreeze and epoch == next_unfreeze[0]:
                self.unfreeze_layers(next_unfreeze[1])
                self.console.log(
                    f"[yellow]Unfroze layers up to {next_unfreeze[1]} at epoch {epoch + 1}."
                )
                next_unfreeze = next(unfreeze_schedule, None)

            train_loss, train_auroc, train_f1 = self.train(epoch)
            val_loss, val_auroc, val_f1 = self.validate(epoch)

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.no_improvement_counter = 0
                torch.save(self.model.state_dict(), MODEL_DIR)
                self.console.log(f"[yellow]Model improved and saved at {MODEL_DIR}")
                if self.use_wandb:
                    wandb.log({"best_val_loss": val_loss})
            else:
                self.no_improvement_counter += 1

            # Early stopping or warning
            if early_stopping and self.no_improvement_counter >= PATIENCE:
                self.console.log(
                    f"[red]Early stopping triggered after {epoch + 1} epochs due to no improvement."
                )
                if self.use_wandb:
                    wandb.log({"early_stopping_epoch": epoch + 1})
                break

            # Learning rate scheduling
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)  # Metric-based scheduler
            else:
                self.scheduler.step()  # Step-based scheduler

        if self.use_wandb:
            wandb.finish()


# Example usage:
# model = YourModel()
# criterion = BCEWithLogitsLoss()
# optimizer = Adam(model.parameters(), lr=1e-3)
# scheduler = StepLR(optimizer, step_size=5, gamma=0
