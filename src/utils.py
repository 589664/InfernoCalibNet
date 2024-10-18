from PIL import Image
import numpy as np

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
    Computes the mean and standard deviation of pixel values for a list of images.

    Args:
        image_paths (list[str]): A list of paths to image files.
        img_size (int): The target size for resizing the images (same value for width and height).

    Returns:
        tuple[float, float]: The mean and standard deviation of the pixel values across all images.
    """
    pixel_values = []
    for img_path in image_paths:
        img = load_image(img_path, img_size)
        pixel_values.append(np.array(img).flatten())
    pixel_values = np.concatenate(pixel_values)
    mean = np.mean(pixel_values)
    std = np.std(pixel_values)
    return mean, std

