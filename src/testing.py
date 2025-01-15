import os
import torch
from PIL import Image
from torchvision import transforms

from src.NNModels import XrayResNet

from config import (
    LR,
    BATCH_SZ,
    NUM_WRKRS,
    IMG_SIZE,
    MEAN,
    STD,
    DISEASE_LABELS,
    XRAY_DIR,
    MODEL_DIR,
    MODEL,
)


def predict_on_image(image_name: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XrayResNet(model_type=MODEL)
    model.load_state_dict(torch.load(MODEL_DIR, weights_only=True))
    model.to(device)
    model.eval()

    # Construct the full image path using XRAY_DIR
    image_path = os.path.join(XRAY_DIR, image_name)
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Define the transformations
    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[MEAN], std=[STD]
            ),  # Replace MEAN and STD with your values
        ]
    )

    # Load and preprocess the image
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Forward pass
    with torch.no_grad():
        outputs = model(image)
        probabilities = (
            torch.sigmoid(outputs).squeeze().tolist()
        )  # Convert logits to probabilities

    # Create a dictionary of predictions with disease names
    predictions = {
        disease: prob for disease, prob in zip(DISEASE_LABELS, probabilities)
    }
    return predictions
