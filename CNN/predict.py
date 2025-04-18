#====================================================================
# 📂 Imports
#====================================================================
# System
import os
import json
import tempfile
import subprocess
from pathlib import Path

# Data Handling
import pandas as pd
import numpy as np

# Machine Learning
import torch
import matplotlib.pyplot as plt
from torchvision import transforms as T
from PIL import Image

# Visualization and Console Output
from rich import print
from rich.console import Console
from rich.table import Table
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Project Specific
from CNN import InfernoCalibNet, CALIB_DIR

console = Console()

#====================================================================
# 📸 Grad-CAM Visualization
#====================================================================
def runGradCAM(model: torch.nn.Module, input_tensor: torch.Tensor, device: torch.device):
    model.eval()
    target_layer = model.base_model[-1]

    cam = GradCAM(model=model, target_layers=[target_layer])
    input_tensor = input_tensor.unsqueeze(0).to(device)
    input_tensor.requires_grad = True

    with torch.no_grad():
        output = model(input_tensor)
    predicted_class = output.argmax(dim=1).item()
    targets = [ClassifierOutputTarget(predicted_class)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    grayscale_cam = np.maximum(grayscale_cam, 0)
    grayscale_cam = grayscale_cam - grayscale_cam.min()
    grayscale_cam = grayscale_cam / grayscale_cam.max()

    img = input_tensor.detach().cpu().squeeze().numpy()
    fig, ax = plt.subplots()
    ax.imshow(img, cmap="gray")
    ax.imshow(grayscale_cam, cmap="jet", alpha=0.5)
    ax.set_title(f"Grad-CAM for predicted class {predicted_class}")
    plt.axis("off")
    plt.show()

#====================================================================
# 🧠 Model Inference Function
#====================================================================
def predict_from_image_path(image_path: str) -> tuple[tuple[float, float], np.ndarray, np.ndarray]:
    """
    Runs prediction on a manually provided image path and returns raw float logits, probabilities, and binary predictions.

    Args:
        image_path (str): Path to the grayscale X-ray image.

    Returns:
        tuple: ((logit1, logit2), probabilities, predictions), logits as a float tuple.
    """
    torch.cuda.empty_cache()

    #====================================================================
    # 📂 Define Input Parameters
    #====================================================================
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.49765], std=[0.22854]),
    ])

    image = Image.open(image_path).convert("L")
    image_tensor = transform(image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InfernoCalibNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(CALIB_DIR / "InfernoCalibNetML.pth", weights_only=True))
    model.eval()

    image_tensor = image_tensor.to(device)

    #====================================================================
    # 🔍 Run Model Prediction
    #====================================================================
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        logits = output.cpu().squeeze().numpy().astype(np.float32)
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        predictions = (probs > 0.5).astype(int)

    #====================================================================
    # 📊 Display Prediction Results
    #====================================================================
    console.rule("[bold green] CNN Prediction Result")
    table = Table(title="Model Output for Manual Input", show_lines=True)
    table.add_column("Label", justify="center")
    table.add_column("Logit", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Prediction", justify="right")

    for label, logit, prob, pred_val in zip(
        ["Effusion", "Atelectasis"], logits, probs, predictions
    ):
        table.add_row(label, f"{logit:.4f}", f"{prob:.4f}", str(int(pred_val)))

    console.print(table)

    #====================================================================
    # 📸 Generate Grad-CAM Visualization
    #====================================================================
    runGradCAM(model, image_tensor, device)

    return (float(logits[0]), float(logits[1])), probs, predictions


def run_inferno_prediction(
    predictor_sets: list[dict[str, float | None]],
    predictand_input: dict[str, list],
    model_path: Path = Path("data/inferno/combined_new_MT/learnt.rds"),
    rscript_path: Path = Path("RScripts/inferno2PY.R"),
    input_csv_path: Path = Path("data/inferno/calibration_test.csv"),
    predictands: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    """
    Runs the Inferno prediction pipeline by preparing input predictors and predictands,
    generating a config file, invoking an R script for prediction, and returning results.

    Args:
        predictor_sets (list[dict[str, float | None]]): List of predictor dictionaries.
        predictand_input (dict[str, list]): Dictionary of predictand values.
        model_path (Path, optional): Path to the R model file.
        rscript_path (Path, optional): Path to the R script file.
        input_csv_path (Path, optional): Path to a dummy CSV input file.
        predictands (list[str] | None): List of predictand names. Defaults to keys of predictand_input.

    Returns:
        tuple[pd.DataFrame, dict]: DataFrame of predictions and the raw result dictionary.
    """
    #====================================================================
    # 📂 Normalize and Prepare Input
    #====================================================================
    all_keys = set().union(*predictor_sets)
    for row in predictor_sets:
        for key in all_keys:
            row.setdefault(key, None)

    predictor_input = {key: [row[key] for row in predictor_sets] for key in all_keys}
    manual_input_values = {**predictor_input, **predictand_input}
    predictors = list(predictor_input.keys())

    if predictands is None:
        predictands = list(predictand_input.keys())

    config = {
        "input_csv": str(input_csv_path),
        "model_path": str(model_path),
        "quantiles": [0.055, 0.25, 0.75, 0.945],
        "input_values": manual_input_values,
        "predictors": predictors,
        "predictands": predictands
    }

    console.rule("[bold yellow]🚀Params for Inferno")
    print(config)

    #====================================================================
    # 🔍 Write Config to Temporary JSON File
    #====================================================================
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as temp_config_file:
        json.dump(config, temp_config_file, indent=2)
        temp_config_path = temp_config_file.name

    #====================================================================
    # ♻️ Run R Script Using Config
    #====================================================================
    subprocess.run(["Rscript", str(rscript_path), temp_config_path], check=True)

    #====================================================================
    # 📂 Load Result from R Output
    #====================================================================
    result_path = os.path.join(os.path.dirname(str(model_path)), "result_probs.json")
    with open(result_path, "r") as f:
        result = json.load(f)

    console.rule("[bold cyan]Inferno Results")

    print("\n📊 Prediction Values:")
    df = pd.DataFrame(result["values"])
    print(df.to_string(index=True))

    #====================================================================
    # 🔧 Clean Up Temporary Files
    #====================================================================
    os.remove(temp_config_path)
    os.remove(result_path)

    return df, result
