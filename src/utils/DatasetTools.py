import os
import pandas as pd
import torch
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

    # Add ImagePath column with full image paths as the last column
    metadata["ImagePath"] = metadata["ImageID"].apply(lambda x: image_paths[x])

    # Keep only the required columns
    metadata = metadata[
        ["ImageID", "LabelsList", "PatientID", "MultiHotLabels", "ImagePath"]
    ]

    # Group by PatientID to ensure no overlap between splits
    patient_ids = metadata["PatientID"].unique()
    torch.manual_seed(RAND_STATE)
    shuffled_ids = torch.randperm(len(patient_ids)).tolist()

    # Calculate split indices
    train_end = int(len(patient_ids) * TRAIN_PCT)
    val_end = train_end + int(len(patient_ids) * VAL_PCT)

    train_ids = patient_ids[shuffled_ids[:train_end]]
    val_ids = patient_ids[shuffled_ids[train_end:val_end]]
    test_ids = patient_ids[shuffled_ids[val_end:]]

    # Split metadata into train, validation, and test sets
    train_metadata = metadata[metadata["PatientID"].isin(train_ids)].reset_index(
        drop=True
    )
    val_metadata = metadata[metadata["PatientID"].isin(val_ids)].reset_index(drop=True)
    test_metadata = metadata[metadata["PatientID"].isin(test_ids)].reset_index(
        drop=True
    )

    # Save the split datasets to CSV files
    os.makedirs(OUT_DIR, exist_ok=True)
    train_csv_path = os.path.join(OUT_DIR, "train.csv")
    val_csv_path = os.path.join(OUT_DIR, "validate.csv")
    test_csv_path = os.path.join(OUT_DIR, "test.csv")

    train_metadata.to_csv(train_csv_path, index=False)
    val_metadata.to_csv(val_csv_path, index=False)
    test_metadata.to_csv(test_csv_path, index=False)

    # Return statistics
    stats = {
        "train_size": len(train_metadata),
        "validate_size": len(val_metadata),
        "test_size": len(test_metadata),
        "total_size": len(metadata),
        "train_csv_path": train_csv_path,
        "validate_csv_path": val_csv_path,
        "test_csv_path": test_csv_path,
    }

    return stats


# Example usage:
# stats = preprocess_and_split_csv()
# print(stats)
