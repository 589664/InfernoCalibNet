import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from config import (
    CSV_PATH,
    XRAY_DIR,
    DISEASE_LABELS,
    TRAIN_PCT,
    VAL_PCT,
    TEST_PCT,
    RAND_STATE,
    OUT_DIR,
)


from sklearn.model_selection import train_test_split


def preprocess_and_split_csv() -> dict:
    # Load the original CSV
    metadata = pd.read_csv(CSV_PATH)

    # Filter rows to include only diseases in DISEASE_LABELS
    metadata = metadata[
        metadata["Finding Labels"].apply(
            lambda labels: any(label in DISEASE_LABELS for label in labels.split("|"))
        )
    ]

    # Map image paths to their presence in the folder
    image_paths = {
        os.path.basename(root_file): os.path.join(root, root_file)
        for root, _, files in os.walk(XRAY_DIR)
        for root_file in files
    }

    # Keep only rows where images exist in the folder
    metadata = metadata[
        metadata["Image Index"].apply(lambda x: x in image_paths)
    ].copy()

    # Rename and keep only the necessary columns
    metadata.rename(
        columns={
            "Image Index": "ImageID",
            "Finding Labels": "Labels",
            "Patient ID": "PatientID",
        },
        inplace=True,
    )

    # Convert labels to a consistent format and generate MultiHotLabels
    metadata["LabelsList"] = metadata["Labels"].apply(lambda labels: labels.split("|"))
    metadata["MultiHotLabels"] = metadata["LabelsList"].apply(
        lambda labels: [1 if disease in labels else 0 for disease in DISEASE_LABELS]
    )

    # Add a StratifyGroup column based on the first 4 characters of the Finding Labels
    metadata["StratifyGroup"] = metadata["Labels"].apply(lambda x: x[:4])

    # Add ImagePath column with full image paths as the last column
    metadata["ImagePath"] = metadata["ImageID"].apply(lambda x: image_paths[x])

    # Keep only the required columns
    metadata = metadata[
        [
            "ImageID",
            "LabelsList",
            "PatientID",
            "MultiHotLabels",
            "ImagePath",
            "StratifyGroup",
        ]
    ]

    # Split metadata into train, validation, and test sets using stratify on StratifyGroup
    train_data, temp_data = train_test_split(
        metadata,
        test_size=1 - TRAIN_PCT,
        stratify=metadata["StratifyGroup"],
        random_state=RAND_STATE,
    )

    val_size_adjusted = VAL_PCT / (
        VAL_PCT + TEST_PCT
    )  # Adjust validation size relative to remaining data

    val_data, test_data = train_test_split(
        temp_data,
        test_size=1 - val_size_adjusted,
        stratify=temp_data["StratifyGroup"],
        random_state=RAND_STATE,
    )

    # Save the split datasets to CSV files
    os.makedirs(OUT_DIR, exist_ok=True)
    train_csv_path = os.path.join(OUT_DIR, "train.csv")
    val_csv_path = os.path.join(OUT_DIR, "validate.csv")
    test_csv_path = os.path.join(OUT_DIR, "test.csv")

    train_data.to_csv(train_csv_path, index=False)
    val_data.to_csv(val_csv_path, index=False)
    test_data.to_csv(test_csv_path, index=False)

    # Return statistics
    stats = {
        "train_size": len(train_data),
        "validate_size": len(val_data),
        "test_size": len(test_data),
        "total_size": len(metadata),
        "train_csv_path": train_csv_path,
        "validate_csv_path": val_csv_path,
        "test_csv_path": test_csv_path,
    }

    return stats


# Example usage:
# stats = preprocess_and_split_csv()
# print(stats)
# __________________________________________________________________________________


def inspect_distribution(split: str) -> dict:
    # Load the CSV
    metadata = pd.read_csv(os.path.join(OUT_DIR, f"{split}.csv"))

    # Ensure MultiHotLabels column exists
    if "MultiHotLabels" not in metadata.columns:
        raise ValueError("The provided CSV does not contain 'MultiHotLabels'.")

    # Convert MultiHotLabels from strings to lists
    metadata["MultiHotLabels"] = metadata["MultiHotLabels"].apply(eval)

    # Initialize counts for each class
    class_counts = {disease: 0 for disease in DISEASE_LABELS}

    # Count occurrences of each class
    for labels in metadata["MultiHotLabels"]:
        for idx, present in enumerate(labels):
            if present:
                class_counts[DISEASE_LABELS[idx]] += 1

    # Calculate total number of samples
    total_samples = len(metadata)

    # Calculate percentages for each class
    class_percentages = {
        disease: (count / total_samples) * 100
        for disease, count in class_counts.items()
    }

    # Combine counts and percentages
    distribution_stats = {
        "class_counts": class_counts,
        "class_percentages": class_percentages,
        "total_samples": total_samples,
    }

    return distribution_stats


# Example usage:
# distribution = inspect_distribution("train")
# print(distribution)
# __________________________________________________________________________________


def calculate_class_weights(split: str) -> tuple:
    metadata = pd.read_csv(os.path.join(OUT_DIR, f"{split}.csv"))

    if "MultiHotLabels" not in metadata.columns:
        raise ValueError("The provided CSV does not contain 'MultiHotLabels'.")

    metadata["MultiHotLabels"] = metadata["MultiHotLabels"].apply(eval)

    # Initialize counts for positives and negatives
    positive_counts = [0] * len(DISEASE_LABELS)
    total_samples = len(metadata)

    for labels in metadata["MultiHotLabels"]:
        for idx, present in enumerate(labels):
            if present:
                positive_counts[idx] += 1

    negative_counts = [total_samples - pos_count for pos_count in positive_counts]

    # Compute weights as the ratio of negatives to positives
    class_weights = [
        neg / (pos + 1e-6) for pos, neg in zip(positive_counts, negative_counts)
    ]

    # Convert to tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # Create a dictionary mapping labels to weights
    weights_dict = {
        DISEASE_LABELS[i]: class_weights[i].item() for i in range(len(DISEASE_LABELS))
    }

    return class_weights, weights_dict


# class_weights, weights_dict = calculate_class_weights("train")
# print(class_weights)
# print(weights_dict)
# __________________________________________________________________________________
