import torch
from torch.utils.data import DataLoader
from tqdm.rich import tqdm
from tqdm import TqdmExperimentalWarning
from torchmetrics.classification import MultilabelAUROC, MultilabelF1Score
from rich.console import Console
from config import EPOCHS, PATIENCE, NUM_CL, MODEL_DIR
import warnings


class ICNTrainer:
    def __init__(
        self,
        device: torch.device,
        model: torch.nn.Module,
        loaders: dict,
        criterion: torch.nn.Module,
        optimizer: torch.optim,
        scheduler: torch.optim,
    ) -> None:
        self.device = device
        self.model = model.to(self.device)
        self.console = Console()

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

        # Suppress TqdmExperimentalWarning
        warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

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

        self.console.log(
            f"[blue]Val Loss: {avg_loss:.4f}, Val AUROC: {val_auroc:.4f}, Val F1: {val_f1:.4f}"
        )
        return avg_loss, val_auroc, val_f1

    def fit(self) -> None:
        for epoch in range(EPOCHS):
            train_loss, train_auroc, train_f1 = self.train(epoch)
            val_loss, val_auroc, val_f1 = self.validate(epoch)

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.no_improvement_counter = 0
                torch.save(self.model.state_dict(), MODEL_DIR)
                self.console.log(f"[yellow]Model improved and saved at {MODEL_DIR}")
            else:
                self.no_improvement_counter += 1

            # Warning for no improvement
            if self.no_improvement_counter >= PATIENCE:
                self.console.log(
                    f"[red]No improvement in validation loss for {PATIENCE} epochs. Consider stopping."
                )

            # Learning rate scheduling
            self.scheduler.step(val_loss)


# Example usage:
# from torch.optim import Adam
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch.nn import BCEWithLogitsLoss
# from your_model_module import YourModel
# from your_dataset_module import train_loader, val_loader
#
# model = YourModel()
# criterion = BCEWithLogitsLoss()
# optimizer = Adam(model.parameters(), lr=1e-3)
# scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.1)
# loaders = {"train": train_loader, "val": val_loader}
#
# trainer = ICNTrainer(device=torch.device("cuda"), model=model, loaders=loaders, criterion=criterion, optimizer=optimizer, scheduler=scheduler)
# trainer.fit()
