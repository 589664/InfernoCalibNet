import torch
import wandb
import config
import warnings
import numpy as np
from tqdm.rich import tqdm
from rich.console import Console
from tqdm import TqdmExperimentalWarning
from sklearn.metrics import roc_auc_score
from torchmetrics.classification import MultilabelF1Score, MultilabelAUROC

epochs = config.EPOCHS
classes = config.NUM_CLASSES
model_dir = config.MODEL_DIR


class ICNTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        project_name,
        config=None,
        log_every_n_batches=10,
    ):
        """
        Initialize the ICNTrainer class with a PyTorch model and W&B integration.
        """
        # Existing Console instance
        self.console = Console()

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.log_every_n_batches = log_every_n_batches

        # Initialize scheduler and W&B
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=0.01,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
        )
        wandb.init(project=project_name, config=config)

    def train_one_epoch(self):
        """Train the model for one epoch with tqdm and log results with rich."""
        self.model.train()
        running_loss = 0.0
        output_list = []
        label_list = []
        # Initialize torchmetrics
        f1_metric = MultilabelF1Score(num_labels=classes, average="macro").to(
            self.device
        )
        auc_metric = MultilabelAUROC(num_labels=classes, average="macro").to(
            self.device
        )

        # Using tqdm.rich for training progress bar
        train_progress = tqdm(self.train_loader, desc="Training", leave=False)

        # Training steps loop
        for batch_idx, (images, labels) in enumerate(train_progress):
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)  # BCEWithLogitsLoss

            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Update running loss
            batch_loss = loss.item()
            running_loss += batch_loss

            # Sigmoid activation for predictions where output is
            # confidence scores/probabilities per image in current batch
            sigmoid_outputs = torch.sigmoid(logits)

            # Update metrics
            f1_metric.update(sigmoid_outputs, labels.long())
            auc_metric.update(sigmoid_outputs, labels.long())

            # Log batch loss to WandB
            if batch_idx % self.log_every_n_batches == 0:
                wandb.log({"batch_loss": batch_loss})

            # Update tqdm bar with average loss
            train_progress.set_postfix(loss=running_loss / (batch_idx + 1))

        # Epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_f1 = f1_metric.compute()  # Compute the F1 score for the epoch
        epoch_auc = auc_metric.compute()  # Compute the AUROC for the epoch

        # Reset metrics for the next epoch
        f1_metric.reset()
        auc_metric.reset()

        # Log metrics to WandB
        wandb.log(
            {
                "epoch_loss": epoch_loss,
                "train_f1": epoch_f1.item(),
                "train_auc": epoch_auc.item(),
            }
        )

        # Log results using the existing console
        self.console.log(
            f"[bold green]Training Loss: {epoch_loss:.4f}, AUC: {epoch_auc:.4f}[/bold green]"
        )
        return epoch_loss, epoch_f1.item(), epoch_auc.item()

    # Inference
    def validate_one_epoch(self):
        """Validate the model for one epoch with tqdm and log results with rich."""
        self.model.eval()
        val_loss = 0.0
        output_list = []
        label_list = []

        # Using tqdm.rich for validation progress bar
        val_progress = tqdm(self.val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for images, labels in val_progress:
                images, labels = images.to(self.device), labels.to(self.device)

                # Validation step
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                # Apply sigmoid to get probabilities for AUC calculation
                sigmoid_outputs = torch.sigmoid(outputs)
                output_list.append(sigmoid_outputs.cpu())
                label_list.append(labels.cpu())

        # Calculate metrics
        val_loss = val_loss / len(self.val_loader)
        outputs_concat = torch.cat(output_list)
        labels_concat = torch.cat(label_list)

        try:
            auc = roc_auc_score(
                labels_concat.cpu().numpy(), outputs_concat.cpu().numpy()
            )
        except ValueError:
            auc = None

        wandb.log({"val_loss": val_loss, "val_auc": auc})

        # Log results using the existing console
        if auc:
            self.console.log(
                f"[bold magenta]Validation Loss: {val_loss:.4f}, AUC: {auc:.4f}[/bold magenta]"
            )
        else:
            self.console.log(
                f"[bold magenta]Validation Loss: {val_loss:.4f}, AUC: N/A[/bold magenta]"
            )

        return val_loss, auc

    def fit(self, epochs, early_stopping_patience=5):
        """Train and validate the model for a specified number of epochs with early stopping."""

        # Supress the warning
        warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

        patience_counter = 0  # Counter for early stopping
        best_model_path = "best_model.pth"
        best_val_loss = float("inf")  # Track the best validation loss

        # Run training & validation
        for epoch in range(epochs):
            self.console.log(f"[bold cyan]Epoch {epoch + 1}/{epochs}[/bold cyan]")

            # Training
            train_loss, train_f1, train_auc = self.train_one_epoch()

            # Validation
            val_loss, val_auc = self.validate_one_epoch()

            # Optionally log or save the model after each epoch
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_f1": train_f1,
                    "train_auc": train_auc,
                    "val_loss": val_loss,
                    # "val_f1": val_f1,
                    "val_auc": val_auc,
                    "learning_rate": self.optimizer.param_groups[0][
                        "lr"
                    ],  # Log learning rate
                }
            )

            # Save the model if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), model_dir / best_model_path)
                self.console.log(
                    f"[bold green]Validation loss improved. Model saved at epoch {epoch + 1}[/bold green]"
                )
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.console.log(
                    f"[bold red]Early stopping triggered at epoch {epoch + 1}[/bold red]"
                )
                break

        self.console.log("[bold cyan]Training complete.[/bold cyan]")
        self.console.log(
            f"[bold green]Best validation loss: {best_val_loss:.4f}[/bold green]"
        )
