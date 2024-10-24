import torch
from tqdm import tqdm
import wandb

# ICNTrainer Class for PyTorch
class ICNTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, project_name, config=None):
        """
        Initialize the ICNTrainer class with a PyTorch model and W&B integration.

        Args:
        - model: The PyTorch model to be trained.
        - train_loader: The DataLoader for training data.
        - val_loader: The DataLoader for validation data.
        - optimizer: The optimizer for training (e.g., Adam, SGD).
        - criterion: The loss function (e.g., BCEWithLogitsLoss).
        - device: The device (e.g., 'cuda' or 'cpu').
        - project_name: The W&B project name.
        - config (optional): Dictionary containing hyperparameters and metadata to track in W&B.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Initialize Weights & Biases (W&B) with optional config
        wandb.init(
            project=project_name,
            config=config  # Passing the config dictionary to W&B if provided
        )

    def train_one_epoch(self):
        """Train the model for one epoch and log results with TQDM and W&B."""
        self.model.train()
        running_loss = 0.0

        # Initialize TQDM progress bar for training
        train_progress = tqdm(self.train_loader, desc="Training", leave=False)

        for images, labels in train_progress:
            images, labels = images.to(self.device), labels.to(self.device)

            # Debugging the shapes
            print(f"Images shape: {images.shape}")   # Expected: [batch_size, 1, 500, 500]
            print(f"Labels shape: {labels.shape}")   # Expected: [batch_size, 15]

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)

            # Debugging the output shape
            print(f"Outputs shape: {outputs.shape}") # Expected: [batch_size, 15]

            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            # Update the TQDM progress bar
            train_progress.set_postfix(loss=running_loss / (train_progress.n + 1))

            # Log the loss for each batch to Weights & Biases
            wandb.log({"batch_loss": loss.item()})

        # Log epoch-level loss
        epoch_loss = running_loss / len(self.train_loader)
        wandb.log({"epoch_loss": epoch_loss})
        print(f"Training Loss: {epoch_loss:.4f}")

        return epoch_loss


    def validate_one_epoch(self):
        """Validate the model for one epoch and log results with TQDM and W&B."""
        self.model.eval()
        val_loss = 0.0

        # Initialize TQDM progress bar for validation
        val_progress = tqdm(self.val_loader, desc="Validation", leave=False)

        with torch.no_grad():
            for images, labels in val_progress:
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

        # Log validation loss to Weights & Biases
        val_loss = val_loss / len(self.val_loader)
        wandb.log({"val_loss": val_loss})
        print(f"Validation Loss: {val_loss:.4f}")

        return val_loss

    def fit(self, epochs):
        """Train and validate the model for the specified number of epochs."""
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            print('-' * 20)

            # Train for one epoch
            train_loss = self.train_one_epoch()

            # Validate for one epoch
            val_loss = self.validate_one_epoch()

            # Optionally, save model checkpoints after each epoch
            torch.save(self.model.state_dict(), f"model_epoch_{epoch + 1}.pth")

        print("Training complete.")
