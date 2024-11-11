import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
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

#########################################################################################################

def calculate_balanced_label_statistics(df: pd.DataFrame, target_percentage: float = 3.5) -> pd.DataFrame:
    """
    Calculates label statistics and determines augmentation factors for underrepresented labels.

    Args:
        df (pd.DataFrame): The preprocessed metadata DataFrame containing 'Labels'.
        target_percentage (float): The target percentage for balancing labels (default: 3.5%).

    Returns:
        pd.DataFrame: A DataFrame with statistics and augmentation factors for each label.
    """
    no_flip_labels = ['Cardiomegaly', 'Pneumothorax']

    # Flatten labels list to count individual occurrences
    all_labels = [label for labels_list in df['Labels'] for label in labels_list]
    label_counts = Counter(all_labels)

    # Calculate total images
    total_images = len(df)

    # Create DataFrame for label statistics
    label_stats = pd.DataFrame({
        'Label': list(label_counts.keys()),
        'Label_Occurrence': list(label_counts.values())
    })

    # Calculate label occurrence percentage
    label_stats['Label_Occurrence_Percentage'] = (label_stats['Label_Occurrence'] / label_stats['Label_Occurrence'].sum()) * 100

    # Calculate augmentation factor based on underrepresentation
    def calculate_augmentation(label, percentage):
        if label in no_flip_labels:
            # For labels that can't be flipped
            return max(1, min(2, int(target_percentage / percentage))) if percentage < target_percentage else 0
        else:
            # For labels that can be flipped
            return max(1, min(5, int(target_percentage / percentage))) if percentage < target_percentage else 0

    label_stats['Augmentation_Factor'] = label_stats.apply(
        lambda row: calculate_augmentation(row['Label'], row['Label_Occurrence_Percentage']), axis=1
    )

    # Drop decimals from Label_Occurrence and Augmentation_Factor
    label_stats['Label_Occurrence'] = label_stats['Label_Occurrence'].astype(int)
    label_stats['Augmentation_Factor'] = label_stats['Augmentation_Factor'].astype(int)

    # Add total columns for sum of label counts and percentages
    label_stats.loc['Total'] = label_stats[['Label_Occurrence', 'Label_Occurrence_Percentage']].sum()
    label_stats.loc['Total', 'Label'] = 'Total'
    label_stats.loc['Total', 'Augmentation_Factor'] = None  # No augmentation for total row

    return label_stats

#########################################################################################################

def distribution_df_split(df: pd.DataFrame, train_size: int, test_size: int):
    """
    Split the dataset while maintaining the distribution of individual labels rather than full combinations.

    Args:
    - df (pd.DataFrame): The full dataset containing the ImageID, Labels, and other metadata.
    - train_size (int): Number of samples to include in the training set.
    - test_size (int): Number of samples to include in the testing/validation set.

    Returns:
    - train_df (pd.DataFrame): Training set.
    - test_df (pd.DataFrame): Testing/validation set.
    """
    # Create a column that reflects the number of individual labels per sample
    df['NumLabels'] = df['Labels'].apply(len)

    # Perform stratified sampling based on the number of labels present in each image
    train_df, test_df = train_test_split(df,
                                        train_size=train_size,
                                        test_size=test_size,
                                        stratify=df['NumLabels'])

    # Drop the helper 'NumLabels' column
    train_df.drop(columns=['NumLabels'], inplace=True)
    test_df.drop(columns=['NumLabels'], inplace=True)

    return train_df, test_df




