# ====================================================================
# 📂 Imports
# ====================================================================
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
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T
from IPython.display import display, SVG

# Visualization and Console Output
from rich import print
import matplotlib.pyplot as plt
from rich.console import Console
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# Project Specific
from CNN import InfernoCalibNet, CALIB_DIR

console = Console()


# ====================================================================
# 📸 Grad-CAM Visualization
# ====================================================================
def runGradCAM(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    predicted_labels: list,
):
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
    ax.imshow(grayscale_cam, cmap="jet", alpha=0.5, rasterized=True)

    title = "Grad-CAM: " + " | ".join(predicted_labels)
    ax.set_title(title, fontname="serif", fontsize=12)
    plt.axis("off")

    plt.show()
    plt.close(fig)


# ====================================================================
# 🧠 Model Inference Function
# ====================================================================
def predict_from_image_path(image_path: str) -> tuple[tuple[float, float], np.ndarray, np.ndarray]:
    torch.cuda.empty_cache()

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.49765], std=[0.22854]),
    ])

    image = Image.open(image_path).convert("L")
    image_tensor = transform(image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InfernoCalibNet(num_classes=2, model_type="resnet50").to(device)
    model.load_state_dict(torch.load(CALIB_DIR / "InfernoCalibNetML50.pth", weights_only=True))
    model.eval()

    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
        logits = output.cpu().squeeze().numpy().astype(np.float32)
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        predictions = (probs > 0.28).astype(int)

    print("[bold green]Prediction Result of CNN")
    labels = ["Effusion", "Atelectasis"]
    predicted_labels = []

    rows = []
    for label, logit, prob, pred_val in zip(labels, logits, probs, predictions):
        if pred_val == 1:
            predicted_labels.append(label)
        rows.append({
            "Label": label,
            "Logit": round(float(logit), 4),
            "Confidence": round(float(prob), 4),
            "Prediction": int(pred_val)
        })

    df_preds = pd.DataFrame(rows)
    print(df_preds.to_string(index=False))

    runGradCAM(model, image_tensor, device, predicted_labels)

    return (float(logits[0]), float(logits[1])), probs, predictions


# ====================================================================
# Inferno prediction
# ====================================================================


def run_inferno_prediction(
    predictor_sets: list[dict[str, float | None]],
    predictand_input: dict[str, list],
    model_path: Path = Path("data/inferno/combinedML50/learnt.rds"),
    rscript_path: Path = Path("RScripts/inferno2PY.R"),
    input_csv_path: Path = Path("data/inferno/calibration_test.csv"),
    predictands: list[str] | None = None
) -> dict:
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
        "input_values": manual_input_values,
        "predictors": predictors,
        "predictands": predictands
    }

    console = Console()
    console.print("\n[bold yellow]🚀Params for Inferno")
    console.print(config)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as temp_config_file:
        json.dump(config, temp_config_file, indent=2)
        temp_config_path = temp_config_file.name

    subprocess.run(["Rscript", str(rscript_path), temp_config_path], check=True)

    result_path = os.path.join(os.path.dirname(str(model_path)), "result_probs.json")
    with open(result_path, "r") as f:
        result = json.load(f)

    os.remove(temp_config_path)
    os.remove(result_path)

    return result
