import numpy as np
import pandas as pd
from PIL import Image
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

