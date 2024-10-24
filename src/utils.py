import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from collections import Counter

def load_image(image_path: str, img_size: int) -> Image.Image:
    """
    Loads an image from the specified path, converts it to grayscale, and resizes it to the specified square size.

    Args:
        image_path (str): The path to the image file.
        img_size (int): The target size for resizing the image (same value for width and height).

    Returns:
        Image.Image: The processed grayscale and resized image.
    """
    img = Image.open(image_path).convert('L')  # Convert to grayscale
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
        img_np = np.array(img).flatten()  # Convert image to a numpy array and flatten it

        # Incrementally update the mean and variance
        n = img_np.size
        n_pixels += n
        mean_sum += np.sum(img_np)
        variance_sum += np.sum((img_np - (mean_sum / n_pixels))**2)

    # Compute the final mean
    mean = mean_sum / n_pixels

    # Compute variance and standard deviation
    variance = variance_sum / n_pixels
    std = np.sqrt(variance)

    return mean, std

#########################################################################################################

def compute_class_weights(df, num_labels):
    """
    Compute class weights for multi-label classification from a DataFrame.

    Args:
    - df (pd.DataFrame): The DataFrame containing the dataset with a 'MultiHotLabels' column.
    - num_labels (int): The number of labels/classes in the dataset (size of the multi-hot vector).

    Returns:
    - class_weights (np.array): Array of computed weights for each label/class.
    """
    # Initialize an array to hold the counts of each label
    label_counts = np.zeros(num_labels)

    # Iterate over the DataFrame and sum the label occurrences
    for labels in df['MultiHotLabels']:
        label_counts += np.array(labels)  # Add the counts of each label

    # Total number of samples
    total_samples = len(df)

    # Compute class weights based on the formula: Total samples / (num_labels * count of each label)
    class_weights = total_samples / (num_labels * label_counts)

    return class_weights

#########################################################################################################

# Custom weighted binary cross-entropy function for PyTorch
def weighted_binary_crossentropy(class_weights):
    """
    Custom weighted binary cross-entropy for multi-label classification.
    """
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')

    def loss_fn(y_true, y_pred):
        # Compute binary cross-entropy with logits
        bce = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none')

        # Apply class weights (you may need to reshape or broadcast the weights here)
        weights = class_weights * y_true + (1 - y_true)
        weighted_bce = weights * bce

        return weighted_bce.mean()

    return loss_fn


#########################################################################################################

def analyze_label_combinations(df: pd.DataFrame, underrepresented_threshold: int = 2) -> pd.DataFrame:
    """
    Analyze the distribution of label combinations and find underrepresented label combinations in the dataset.
    Return a DataFrame summarizing underrepresented combinations and the corresponding images.

    Args:
    - df (pd.DataFrame): Input DataFrame with 'MultiHotLabels' as multi-hot encoded arrays and 'ImageID'.
    - underrepresented_threshold (int): Minimum number of instances for a label combination to not be considered underrepresented.

    Returns:
    - pd.DataFrame: A DataFrame summarizing label combination statistics and the images associated with underrepresented combinations.
    """

    # Extract relevant columns
    label_combinations = df['Labels'].apply(lambda x: sorted(x))

    # Count occurrences of each unique label combination
    combination_counts = Counter(label_combinations.apply(tuple))
    combination_stats = pd.DataFrame([(list(k), v) for k, v in combination_counts.items()], columns=['Label Combination', 'Count'])

    # Identify underrepresented combinations
    underrepresented_combinations = combination_stats[combination_stats['Count'] < underrepresented_threshold]

    # Create a DataFrame to store image associations with underrepresented label combinations
    images_by_combination = pd.DataFrame(columns=['Label Combination', 'ImageID'])

    for combination in underrepresented_combinations['Label Combination']:
        # Find all images associated with the underrepresented combination
        matching_images = df[df['Labels'].apply(lambda x: sorted(x) == combination)]['ImageID']

        # Create a DataFrame of the underrepresented combination and its associated images
        combination_image_df = pd.DataFrame({
            'Label Combination': [combination] * len(matching_images),
            'ImageID': matching_images
        })

        # Append to the main DataFrame
        images_by_combination = pd.concat([images_by_combination, combination_image_df], ignore_index=True)

    # Return statistics of combinations and the underrepresented combinations with corresponding images
    return combination_stats, images_by_combination

