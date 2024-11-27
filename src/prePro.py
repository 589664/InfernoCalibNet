import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.utils import resample
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def preprocess_metadata(
    csv_path: str, image_folder: str, save_path: str
) -> pd.DataFrame:
    """
    Preprocesses metadata from a CSV file and saves the result to another CSV file.
    Filters rows based on the images present in the given folder and stores the correct image paths.

    Args:
        csv_path (str): The path to the input CSV file containing metadata.
        image_folder (str): The path to the folder containing the images and subdirectories.
        save_path (str): The path where the preprocessed metadata will be saved.

    Returns:
        pd.DataFrame: The preprocessed metadata as a pandas DataFrame.
    """
    # Load and filter metadata
    df = pd.read_csv(csv_path)
    filtered_df = df[
        [
            "Image Index",
            "Finding Labels",
            "Patient Age",
            "Patient Gender",
            "View Position",
        ]
    ]
    filtered_df.columns = ["ImageID", "Labels", "Age", "Gender", "XrayView"]

    # Build a mapping of ImageID to full path from image_folder and its subdirectories
    image_id_to_path = {
        os.path.splitext(file)[0]: os.path.join(root, file)
        for root, _, files in os.walk(image_folder)
        for file in files
        if file.endswith(".png")
    }

    # Clean the ImageID column (removing the .png extension if needed)
    filtered_df.loc[:, "ImageID"] = filtered_df["ImageID"].str.replace(
        ".png", "", regex=False
    )

    # Filter DataFrame to include only rows with matching images and add their full paths
    filtered_df = filtered_df[filtered_df["ImageID"].isin(image_id_to_path.keys())]
    filtered_df["ImagePath"] = filtered_df["ImageID"].map(image_id_to_path)

    # Split Labels and binarize them
    filtered_df["Labels"] = filtered_df["Labels"].apply(lambda x: x.split("|"))
    mlb = MultiLabelBinarizer()
    filtered_df["MultiHotLabels"] = mlb.fit_transform(filtered_df["Labels"]).tolist()

    # Convert Gender and XrayView to numerical values
    filtered_df["Gender"] = filtered_df["Gender"].map({"M": 0, "F": 1})
    filtered_df["XrayView"] = filtered_df["XrayView"].map({"PA": 0, "AP": 1})

    # Normalize Age
    filtered_df["Age"] = (filtered_df["Age"] - filtered_df["Age"].mean()) / filtered_df[
        "Age"
    ].std()

    # Save preprocessed metadata
    filtered_df.to_csv(save_path, index=False)

    return filtered_df


#########################################################################################################


def calculate_balanced_label_statistics(
    df: pd.DataFrame, target_percentage: float = 3.5
) -> pd.DataFrame:
    """
    Calculates label statistics and determines augmentation factors for underrepresented labels.

    Args:
        df (pd.DataFrame): The preprocessed metadata DataFrame containing 'Labels'.
        target_percentage (float): The target percentage for balancing labels (default: 3.5%).

    Returns:
        pd.DataFrame: A DataFrame with statistics and augmentation factors for each label.
    """
    no_flip_labels = ["Cardiomegaly", "Pneumothorax"]

    # Flatten labels list to count individual occurrences
    all_labels = [label for labels_list in df["Labels"] for label in labels_list]
    label_counts = Counter(all_labels)

    # Calculate total images
    total_images = len(df)

    # Create DataFrame for label statistics
    label_stats = pd.DataFrame(
        {
            "Label": list(label_counts.keys()),
            "Label_Occurrence": list(label_counts.values()),
        }
    )

    # Calculate label occurrence percentage
    label_stats["Label_Occurrence_Percentage"] = (
        label_stats["Label_Occurrence"] / label_stats["Label_Occurrence"].sum()
    ) * 100

    # Calculate augmentation factor based on underrepresentation
    def calculate_augmentation(label, percentage):
        if label in no_flip_labels:
            # For labels that can't be flipped
            return (
                max(1, min(2, int(target_percentage / percentage)))
                if percentage < target_percentage
                else 0
            )
        else:
            # For labels that can be flipped
            return (
                max(1, min(5, int(target_percentage / percentage)))
                if percentage < target_percentage
                else 0
            )

    label_stats["Augmentation_Factor"] = label_stats.apply(
        lambda row: calculate_augmentation(
            row["Label"], row["Label_Occurrence_Percentage"]
        ),
        axis=1,
    )

    # Drop decimals from Label_Occurrence and Augmentation_Factor
    label_stats["Label_Occurrence"] = label_stats["Label_Occurrence"].astype(int)
    label_stats["Augmentation_Factor"] = label_stats["Augmentation_Factor"].astype(int)

    # Add total columns for sum of label counts and percentages
    label_stats.loc["Total"] = label_stats[
        ["Label_Occurrence", "Label_Occurrence_Percentage"]
    ].sum()
    label_stats.loc["Total", "Label"] = "Total"
    label_stats.loc["Total", "Augmentation_Factor"] = (
        None  # No augmentation for total row
    )

    return label_stats


#########################################################################################################


def split_data(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    no_finding_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # Check if the sum of the ratios is equal to 1.0
    if not abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5:
        raise ValueError("Ratios must sum to 1.0")

    # Separate 'No Finding' labels from others
    no_find = df[df["Labels"].apply(lambda x: "No Finding" in x)].reset_index(drop=True)
    others = df[df["Labels"].apply(lambda x: "No Finding" not in x)].reset_index(
        drop=True
    )

    # Reduce the number of 'No Finding' samples to the target count
    target_count = int(len(df) * no_finding_ratio)
    reduced_no_find = (
        no_find.sample(n=target_count, random_state=42)
        if len(no_find) > target_count
        else no_find
    )

    # Combine the reduced 'No Finding' set with the other labels
    balanced_df = pd.concat([reduced_no_find, others]).reset_index(drop=True)

    # Create a list of all unique labels in the dataset
    labels = list(set([lbl for lbl_list in balanced_df["Labels"] for lbl in lbl_list]))
    # Convert the labels to a multilabel binary format for stratification
    multilabels = pd.DataFrame(
        [{lbl: (lbl in row) for lbl in labels} for row in balanced_df["Labels"]]
    )

    # Split the data into training and temporary sets (for validation and test)
    msss = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=(val_ratio + test_ratio), random_state=42
    )
    train_idx, temp_idx = next(msss.split(balanced_df, multilabels))
    train_df = balanced_df.iloc[train_idx].reset_index(drop=True)
    temp_df = balanced_df.iloc[temp_idx].reset_index(drop=True)
    temp_multilabels = multilabels.iloc[temp_idx].reset_index(drop=True)

    # Split the temporary set into validation and test sets
    val_size = val_ratio / (val_ratio + test_ratio)
    msss_temp = MultilabelStratifiedShuffleSplit(
        n_splits=1, test_size=(1 - val_size), random_state=42
    )
    val_idx, test_idx = next(msss_temp.split(temp_df, temp_multilabels))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df
