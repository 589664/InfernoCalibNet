import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

def preprocess_metadata(csv_path: str, image_folder: str, save_path: str) -> pd.DataFrame:
    """
    Preprocesses metadata from a CSV file and saves the result to another CSV file. Filters rows based on the images present in the given folder.

    Args:
        csv_path (str): The path to the input CSV file containing metadata.
        image_folder (str): The path to the folder containing the images.
        save_path (str): The path where the preprocessed metadata will be saved.

    Returns:
        pd.DataFrame: The preprocessed metadata as a pandas DataFrame.
    """
    # Load and filter metadata
    df = pd.read_csv(csv_path)
    filtered_df = df[['Image Index', 'Finding Labels', 'Patient Age', 'Patient Gender', 'View Position']]
    filtered_df.columns = ['ImageID', 'Labels', 'Age', 'Gender', 'XrayView']

    # Get list of image names in the folder (without extensions)
    image_files = {os.path.splitext(file)[0] for file in os.listdir(image_folder) if file.endswith('.png')}

    # Clean the ImageID column (removing the .png extension if needed)
    # Make an explicit copy of the DataFrame to avoid SettingWithCopyWarning
    filtered_df.loc[:, 'ImageID'] = filtered_df['ImageID'].str.replace('.png', '', regex=False)

    # Filter the DataFrame to include only rows with image names found in the folder
    filtered_df = filtered_df[filtered_df['ImageID'].isin(image_files)]

    # Split Labels and binarize them
    filtered_df['Labels'] = filtered_df['Labels'].apply(lambda x: x.split('|'))
    mlb = MultiLabelBinarizer()
    filtered_df['MultiHotLabels'] = mlb.fit_transform(filtered_df['Labels']).tolist()

    # Convert Gender and XrayView to numerical values
    filtered_df['Gender'] = filtered_df['Gender'].map({'M': 0, 'F': 1})
    filtered_df['XrayView'] = filtered_df['XrayView'].map({'PA': 0, 'AP': 1})

    # Normalize Age
    filtered_df['Age'] = (filtered_df['Age'] - filtered_df['Age'].mean()) / filtered_df['Age'].std()

    # Save preprocessed metadata
    filtered_df.to_csv(save_path, index=False)

    return filtered_df


def calculate_label_statistics(df: pd.DataFrame, target_percentage: float = 3.5) -> pd.DataFrame:
    """
    Calculates label statistics and determines augmentation factors for underrepresented labels,
    with special cases for certain labels (like Cardiomegaly and Pneumothorax) that cannot be flipped.

    Args:
        df (pd.DataFrame): The preprocessed metadata DataFrame containing 'Labels'.
        target_percentage (float): The target percentage for balancing labels (default: 10%).

    Returns:
        pd.DataFrame: A DataFrame with statistics and augmentation factors for each label.
    """
    # Define labels that should not be flipped
    no_flip_labels = ['Cardiomegaly', 'Pneumothorax']

    # Flatten the list of labels to count individual occurrences
    all_labels = [label for labels_list in df['Labels'] for label in labels_list]

    # Use Counter to get the frequency of each label
    label_counts = Counter(all_labels)

    # Calculate total number of images
    total_images = len(df)

    # Create a DataFrame to store the statistics
    label_stats = pd.DataFrame({
        'Label': list(label_counts.keys()),
        'Count': list(label_counts.values())
    })

    # Calculate percentage for each label
    label_stats['Percentage'] = (label_stats['Count'] / total_images) * 100

    # Calculate the augmentation factor for each label
    def calculate_augmentation(label, percentage):
        if label in no_flip_labels:
            # For labels that can't be flipped
            return max(1, min(2, int(target_percentage / percentage))) if percentage < target_percentage else 0
        else:
            # For labels that can be flipped
            return max(1, min(5, int(target_percentage / percentage))) if percentage < target_percentage else 0

    label_stats['Augmentation_Factor'] = label_stats.apply(
        lambda row: calculate_augmentation(row['Label'], row['Percentage']), axis=1
    )

    # Sort by label count (descending)
    label_stats = label_stats.sort_values(by='Count', ascending=False).reset_index(drop=True)

    # Print the statistics
    print(f'Total number of images: {total_images}')
    print('Label statistics:')
    print(label_stats)

    return label_stats
