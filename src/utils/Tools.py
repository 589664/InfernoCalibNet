import os
import pandas as pd
from PIL import Image
from tqdm import tqdm
from typing import Tuple
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


# torch
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from config import (
    IMG_SIZE,
    XRAY_DIR,
    RAND_STATE,
    STRATIFY_COL,
    TRAIN_PCT,
    VAL_PCT,
    TEST_PCT,
    CSV_PATH,
    OUT_DIR,
    DISEASE_LABELS,
)


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.image_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(root_dir)
            for file in files
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  # Dummy label since we're only interested in the images


def compute_mean_std_rgb_dataset(
    root_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    device: torch.device = None,
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(
        ImageFolderDataset(root_dir, transform=transforms.ToTensor()),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    mean = torch.zeros(3, device=device)
    std = torch.zeros(3, device=device)
    total_pixels = 0

    progress_bar = tqdm(loader, desc="Computing mean and std", unit="batch")
    for images, _ in progress_bar:
        images = (
            images.to(device).float().view(images.size(0), 3, -1)
        )  # Flatten to (B, 3, H*W)
        batch_pixels = images.size(0) * images.size(2)
        batch_mean = images.mean(dim=[0, 2])
        batch_std = images.std(dim=[0, 2])

        mean = (mean * total_pixels + batch_mean * batch_pixels) / (
            total_pixels + batch_pixels
        )
        std = (std * total_pixels + batch_std * batch_pixels) / (
            total_pixels + batch_pixels
        )

        total_pixels += batch_pixels

    # Log the final mean and std values to tqdm
    tqdm.write(
        f"Final Mean - R: {mean[0].item():.4f}, G: {mean[1].item():.4f}, B: {mean[2].item():.4f}"
    )
    tqdm.write(
        f"Final Std - R: {std[0].item():.4f}, G: {std[1].item():.4f}, B: {std[2].item():.4f}"
    )

    return (
        (mean[0].item(), std[0].item()),
        (mean[1].item(), std[1].item()),
        (mean[2].item(), std[2].item()),
    )


# __________________________________________________________________________________________


def load_image(image_id: str, image_size: int = IMG_SIZE) -> Image.Image:
    img_path: str = os.path.join(XRAY_DIR, image_id)
    image: Image.Image = Image.open(img_path).convert("RGB")  # Ensure image is RGB
    resized_image: Image.Image = image.resize((image_size, image_size), Image.LANCZOS)
    return resized_image


# __________________________________________________________________________________________


def dataframe_inspector(
    df: pd.DataFrame, column_name: str, disease_classes: list[str]
) -> pd.DataFrame:
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    # Convert the given column into a list of lists with each disease name extracted
    labels_list: list[list[str]] = (
        df[column_name].apply(lambda x: eval(x) if isinstance(x, str) else x).tolist()
    )
    labels_df: pd.DataFrame = pd.DataFrame(0, index=df.index, columns=disease_classes)

    # Populate labels_df with 1s for each disease present in the corresponding row
    for i, diseases in enumerate(labels_list):
        if isinstance(diseases, list):
            for disease in diseases:
                if disease in disease_classes:
                    labels_df.at[i, disease] = 1

    # Initialize the summary DataFrame
    summary_df: pd.DataFrame = pd.DataFrame(
        columns=["Disease", "Value", "Count", "Percentage"]
    )

    # Calculate value counts and percentages for each class and append to summary DataFrame
    for disease in labels_df.columns:
        value_counts: pd.Series = labels_df[disease].value_counts()
        percentages: pd.Series = labels_df[disease].value_counts(normalize=True) * 100

        stats_df: pd.DataFrame = pd.DataFrame(
            {
                "Disease": [disease] * len(value_counts),
                "Value": value_counts.index,
                "Count": value_counts.values,
                "Percentage": percentages.values,
            }
        )

        summary_df = pd.concat([summary_df, stats_df], ignore_index=True)

    return summary_df


# __________________________________________________________________________________________


def split_and_save_dataframe(
    csv_path: str = CSV_PATH,
    out_dir: str = OUT_DIR,
    train_pct: float = TRAIN_PCT,
    val_pct: float = VAL_PCT,
    test_pct: float = TEST_PCT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Load the dataset
    data: pd.DataFrame = pd.read_csv(csv_path)

    # Ensure all disease labels are present as separate columns in the dataframe
    for disease in DISEASE_LABELS:
        if disease not in data.columns:
            data[disease] = data[STRATIFY_COL].apply(lambda x: 1 if disease in x else 0)

    # Create output directory if it does not exist
    os.makedirs(out_dir, exist_ok=True)

    # Split data into training and temporary (validation + test) sets
    labels = data[DISEASE_LABELS].values
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=(1 - train_pct), random_state=RAND_STATE
    )
    train_index, temp_index = next(msss.split(data, labels))
    train_data = data.iloc[train_index]
    temp_data = data.iloc[temp_index]

    # Split temporary set into validation and test sets
    val_ratio = val_pct / (val_pct + test_pct)
    msss_val_test = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=(1 - val_ratio), random_state=RAND_STATE
    )
    val_index, test_index = next(
        msss_val_test.split(temp_data, temp_data[DISEASE_LABELS].values)
    )
    val_data = temp_data.iloc[val_index]
    test_data = temp_data.iloc[test_index]

    # Save datasets to CSV files
    train_data.to_csv(f"{out_dir}/train.csv", index=False)
    val_data.to_csv(f"{out_dir}/val.csv", index=False)
    test_data.to_csv(f"{out_dir}/test.csv", index=False)

    # Print disease distribution statistics for each dataset
    print("Training set disease distribution:")
    for disease, count in zip(DISEASE_LABELS, train_data[DISEASE_LABELS].sum()):
        print(f"  {disease}: {count}")
    print("Validation set disease distribution:")
    for disease, count in zip(DISEASE_LABELS, val_data[DISEASE_LABELS].sum()):
        print(f"  {disease}: {count}")
    print("Test set disease distribution:")
    for disease, count in zip(DISEASE_LABELS, test_data[DISEASE_LABELS].sum()):
        print(f"  {disease}: {count}")

    return train_data, val_data, test_data


# Example of usage:
# train_data, val_data, test_data = split_and_save_dataframe()  # Returns train, val, test DataFrames
# train_data, val_data, test_data = split_and_save_dataframe(csv_path='data/input.csv', out_dir='data/split_output')

# __________________________________________________________________________________________


def count_images_per_class(txt_path: str) -> dict:
    # Load the CSV and TXT files
    data = pd.read_csv(CSV_PATH)
    with open(txt_path, "r") as file:
        image_names = file.read().splitlines()

    # Filter the dataframe to only include rows with image names in the TXT file
    filtered_data = data[data["Image Index"].isin(image_names)]

    # Count the number of images representing each disease based on the STRATIFY_COL
    disease_counts = {
        disease: filtered_data[STRATIFY_COL]
        .apply(lambda x: 1 if disease in x else 0)
        .sum()
        for disease in DISEASE_LABELS
    }

    # Print the disease counts
    print("Image representation per class:")
    for disease, count in disease_counts.items():
        print(f"  {disease}: {count}")

    return disease_counts


# disease_counts = count_images_per_class(txt_path='data/image_list.txt')
# __________________________________________________________________________________________
