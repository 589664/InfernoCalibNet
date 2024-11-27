import numpy as np
import pandas as pd
from PIL import Image


def load_image(image_path: str, img_size: int) -> Image.Image:
    """
    Loads an image from the specified path, converts it to grayscale, and resizes it to the specified square size.

    Args:
        image_path (str): The path to the image file.
        img_size (int): The target size for resizing the image (same value for width and height).

    Returns:
        Image.Image: The processed grayscale and resized image.
    """
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img_resized = img.resize((img_size, img_size))  # Resize to square of size img_size
    return img_resized


def compute_mean_std(image_paths: list[str], img_size: int) -> tuple[float, float]:
    """
    Computes the mean and standard deviation of pixel values for a list of images in a memory-efficient way.
    Args:
        image_paths (list[str]): A list of paths to image files.
        img_size (int): The target size for resizing the images (same value for width and height).
    Returns:
        tuple[float, float]: The mean and standard deviation of the pixel values across all images.
    """
    n_pixels = 0
    mean_sum = 0.0
    variance_sum = 0.0

    for img_path in image_paths:
        img = load_image(img_path, img_size)
        img_np = np.array(
            img
        ).flatten()  # Convert image to a numpy array and flatten it

        # Incrementally update the mean and variance
        n = img_np.size
        n_pixels += n
        mean_sum += np.sum(img_np)
        variance_sum += np.sum((img_np - (mean_sum / n_pixels)) ** 2)

    # Compute the final mean
    mean = mean_sum / n_pixels

    # Compute variance and standard deviation
    variance = variance_sum / n_pixels
    std = np.sqrt(variance)

    return mean, std


#########################################################################################################


def compute_class_weights(df: pd.DataFrame, disease_classes: dict):
    """
    Compute class weights for multi-label classification using multi-hot labels.

    Args:
    - df (pd.DataFrame): DataFrame containing multi-hot labels in the 'MultiHotLabels' column.
    - disease_classes (dict): Dictionary containing disease class names and their target ratios.

    Returns:
    - class_weights (torch.Tensor): Tensor of weights for each class in the format expected by BCEWithLogitsLoss.
    - weights_by_label (dict): Dictionary of weights mapped to each disease label.
    - label_counts (dict): Dictionary containing positive and negative counts for each disease label.
    """
    import torch

    # Extract the multi-hot labels
    labels = np.array(df["MultiHotLabels"].tolist())

    # Total number of samples
    total_samples = len(df)

    # Calculate the positive counts for each class
    positive_counts = labels.sum(axis=0)

    # Calculate the negative counts for each class
    negative_counts = total_samples - positive_counts

    # Calculate the weights for each class
    class_weights = [
        (
            float(negative_counts[i] / positive_counts[i])
            if positive_counts[i] > 0
            else 1.0
        )
        for i in range(len(positive_counts))
    ]

    # Convert the class weights to a torch tensor for BCEWithLogitsLoss
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # Create a dictionary with the disease class names and their corresponding weights
    weights_by_label = {
        label.lower(): class_weights[idx]
        for idx, label in enumerate(disease_classes.keys())
    }

    # Create a dictionary with the positive and negative counts for each disease label
    label_counts = {
        label.lower(): {
            "positive_count": int(positive_counts[idx]),
            "negative_count": int(negative_counts[idx]),
        }
        for idx, label in enumerate(disease_classes.keys())
    }

    return class_weights_tensor, weights_by_label, label_counts


#########################################################################################################


def print_label_statistics(df: pd.DataFrame, all_labels: list):
    """
    Print the statistics of label representation in percentage for a given DataFrame.

    Args:
    - df (pd.DataFrame): DataFrame containing images and labels.
    - all_labels (list): List of all possible labels in the dataset.
    """
    total_count = len(df)
    label_counts = {label: 0 for label in all_labels}

    for labels in df["Labels"]:
        for label in labels:
            if label in label_counts:
                label_counts[label] += 1

    print(f"{'Label':<20} {'Count':<10} {'Percentage (%)':<15}")
    print("-" * 50)
    for label, count in label_counts.items():
        percentage = (count / total_count) * 100
        print(f"{label:<20} {count:<10} {percentage:<15.2f}")
