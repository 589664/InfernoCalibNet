import torch
import wandb
import config
import warnings
import numpy as np
from tqdm.rich import tqdm
from rich.console import Console
from tqdm import TqdmExperimentalWarning
from sklearn.metrics import roc_auc_score

epochs = config.EPOCHS


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

        # Using tqdm.rich for training progress bar
        train_progress = tqdm(self.train_loader, desc="Training", leave=False)

        for batch_idx, (images, labels) in enumerate(train_progress):
            images, labels = images.to(self.device), labels.to(self.device)

            # Training step
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Track batch loss
            batch_loss = loss.item()
            running_loss += batch_loss
            if batch_idx % self.log_every_n_batches == 0:
                wandb.log({"batch_loss": batch_loss})

            # Update tqdm bar with average loss
            train_progress.set_postfix(loss=running_loss / (batch_idx + 1))

            # Collect data for AUC - apply sigmoid for probability scores
            sigmoid_outputs = torch.sigmoid(outputs)
            output_list.extend(sigmoid_outputs.detach().cpu().numpy())
            label_list.extend(labels.detach().cpu().numpy())

        # Epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_auc = roc_auc_score(np.array(label_list), np.array(output_list))
        wandb.log({"epoch_loss": epoch_loss, "train_auc": epoch_auc})

        # Log results using the existing console
        self.console.log(
            f"[bold green]Training Loss: {epoch_loss:.4f}, AUC: {epoch_auc:.4f}[/bold green]"
        )
        return epoch_loss, epoch_auc

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

    def fit(self, epochs):
        # Supress the warning
        warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
        """Train and validate the model for a specified number of epochs."""
        # Run training & validation
        for epoch in range(epochs):
            self.console.log(f"[bold cyan]Epoch {epoch + 1}/{epochs}[/bold cyan]")
            train_loss, train_auc = self.train_one_epoch()
            val_loss, val_auc = self.validate_one_epoch()

            # Optionally log or save the model after each epoch
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_auc": train_auc,
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                }
            )

            # Save model checkpoint if desired
            torch.save(self.model.state_dict(), f"model_epoch_{epoch + 1}.pth")

        self.console.log("[bold cyan]Training complete.[/bold cyan]")
