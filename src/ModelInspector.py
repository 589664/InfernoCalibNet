import torch
from torchvision import transforms
from .utils import load_image  # Importing load_image directly from utils


class ModelInspector:
    def __init__(self, config):
        """
        Initialize the ModelInspector class with the given configuration.

        Args:
            config (dict): Configuration dictionary containing:
                - 'model' (torch.nn.Module): The already modified and trained model.
                - 'model_path' (str): Path to the saved model weights (.pth file).
                - 'input_size' (int): Input image size.
                - 'mean' (float): Mean value for normalization (since it's grayscale, a single value).
                - 'std' (float): Standard deviation value for normalization (single value).
        """
        # Extract config values
        self.model = config.get("model")
        self.model_path = config.get("model_path")
        self.input_size = config.get("input_size")
        self.mean = config.get("mean", 0.5)
        self.std = config.get("std", 0.5)

        # Load model weights
        self.load_model()

    def load_model(self):
        """Load the model weights from the specified path."""
        # Load the state dictionary and apply to the model
        state_dict = torch.load(
            self.model_path, map_location=torch.device("cpu"), weights_only=True
        )
        self.model.load_state_dict(
            state_dict, strict=True
        )  # Load weights strictly assuming matched model structure
        self.model = self.model.cpu()  # Explicitly move model to CPU
        self.model.eval()  # Set the model to evaluation mode

    def get_class_weights_and_biases(self):
        """
        Get the weights and biases of the final classifier layer.

        Returns:
            tuple: (weights, biases)
        """
        # Use state_dict to directly access the final layer weights and biases
        state_dict = self.model.state_dict()

        # Based on the keys you provided, we need to access 'classifier.1.weight' and 'classifier.1.bias'
        weight_key = "classifier.1.weight"
        bias_key = "classifier.1.bias"

        # Extract weights and biases
        if weight_key in state_dict and bias_key in state_dict:
            weights = state_dict[weight_key].cpu().numpy()
            biases = state_dict[bias_key].cpu().numpy()
            return weights, biases
        else:
            raise AttributeError(
                "The final layer weights and biases could not be found. Check the model's state_dict keys."
            )

    def preprocess_image(self, image_path):
        """
        Preprocess an input image to prepare it for prediction.
        Uses load_image from utils.py to load and resize the image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        # Use load_image from utils.py to get the resized grayscale image
        img = load_image(image_path, self.input_size)

        # Convert image to a tensor and normalize it
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.mean], std=[self.std]),
            ]
        )
        input_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        return input_tensor

    def predict(self, image_path):
        """
        Make a prediction for a given image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            dict: Dictionary with class probabilities.
        """
        # Preprocess the image
        input_tensor = self.preprocess_image(image_path)

        class_names = [
            "Atelectasis"
            "Cardiomegaly"
            "Consolidation"
            "Edema"
            "Effusion"
            "Emphysema"
            "Fibrosis"
            "Hernia"
            "Infiltration"
            "Mass"
            "No Finding"
            "Nodule"
            "Pleural_Thickening"
            "Pneumonia"
            "Pneumothorax"
        ]

        # Make predictions
        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()

        # Define the correct class names (adjust accordingly based on your 15 classes)
        class_names = [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Effusion",
            "Emphysema",
            "Fibrosis",
            "Hernia",
            "Infiltration",
            "Mass",
            "No Finding",
            "Nodule",
            "Pleural_Thickening",
            "Pneumonia",
            "Pneumothorax",
        ]

        # Create a dictionary with class names and convert np.float32 to regular float
        predictions = {
            class_name: float(prob)
            for class_name, prob in zip(class_names, probabilities[0])
        }

        # Sort the predictions by probability in descending order
        sorted_predictions = dict(
            sorted(predictions.items(), key=lambda item: item[1], reverse=True)
        )

        return sorted_predictions
