import tensorflow as tf
from tqdm import tqdm
import wandb

# ICNTrainer Class
class ICNTrainer:
    def __init__(self, model, train_dataset, val_dataset, batch_size, project_name):
        """
        Initialize the ModelTrainer class with a compiled model.

        Args:
        - model: The compiled TensorFlow/Keras model to be trained.
        - train_dataset: The dataset for training (e.g., a TensorFlow Dataset object).
        - val_dataset: The dataset for validation (e.g., a TensorFlow Dataset object).
        - batch_size: The batch size for training.
        - project_name: The W&B project name.
        - entity: The W&B entity or username.
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

        # Initialize Weights & Biases (W&B)
        wandb.init(project=project_name)

    def train_one_epoch(self):
        """Train the model for one epoch and log results with TQDM and W&B."""
        running_loss = 0.0
        train_steps_per_epoch = len(self.train_dataset) // self.batch_size

        # Initialize TQDM progress bar for training
        train_progress = tqdm(self.train_dataset, total=train_steps_per_epoch, desc="Training", leave=False)

        for step, (images, labels) in enumerate(train_progress):
            with tf.GradientTape() as tape:
                predictions = self.model(images, training=True)
                loss = self.model.compiled_loss(labels, predictions)

            gradients = tape.gradient(loss, self.model.trainable_weights)
            self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

            running_loss += loss.numpy()

            # Update the TQDM progress bar
            train_progress.set_postfix(loss=running_loss / (step + 1))

            # Log the loss for each batch to Weights & Biases
            wandb.log({"batch_loss": loss.numpy()})

        # Log epoch-level loss
        epoch_loss = running_loss / train_steps_per_epoch
        wandb.log({"epoch_loss": epoch_loss})
        print(f"Training Loss: {epoch_loss:.4f}")

        return epoch_loss

    def validate_one_epoch(self):
        """Validate the model for one epoch and log results with TQDM and W&B."""
        val_loss = 0.0
        val_steps_per_epoch = len(self.val_dataset) // self.batch_size

        # Initialize TQDM progress bar for validation
        val_progress = tqdm(self.val_dataset, total=val_steps_per_epoch, desc="Validation", leave=False)

        for step, (images, labels) in enumerate(val_progress):
            predictions = self.model(images, training=False)
            loss = self.model.compiled_loss(labels, predictions)
            val_loss += loss.numpy()

        # Log validation loss to Weights & Biases
        val_loss = val_loss / val_steps_per_epoch
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
            # self.model.save(f"model_epoch_{epoch+1}.h5")

        print("Training complete.")
