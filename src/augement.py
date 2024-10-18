import os
import numpy as np
from PIL import Image
from torchvision import transforms
from .utils import load_image

# Define consistent augmentations (flip, rotate left, rotate right)
augmentations = {
    'rotate_left': transforms.RandomRotation(degrees=(-10, -10)),
    'rotate_right': transforms.RandomRotation(degrees=(10, 10)),
    'flip': transforms.RandomHorizontalFlip(p=1.0),  # Always flip horizontally
}

# Function to save augmented images based on calculated augmentation factors and labels
def save_augmented_images(img_path, save_dir, image_id, disease_labels, augmentation_factor):
    """
    Apply augmentations to an image based on its disease labels and augmentation factor.

    Args:
        img_path (str): Path to the original image.
        save_dir (str): Directory where augmented images will be saved.
        image_id (str): Unique identifier for the image.
        disease_labels (list): List of disease labels associated with the image.
        augmentation_factor (int): The number of augmentations to apply.
    """
    image = load_image(img_path, 300)

    # Skip augmentation if factor is 0
    if augmentation_factor == 0:
        print(f'Skipping augmentation for image {image_id} (Factor = 0)')
        return

    applied_augmentations = []

    # For factor 1 or more, apply one random rotation (left or right)
    if augmentation_factor >= 1:
        if np.random.rand() > 0.5:
            rotated_image = augmentations['rotate_left'](image)
            applied_augmentations.append('rotate_left')
            rotated_image.save(f'{save_dir}/{image_id}_rotate_left.png')
        else:
            rotated_image = augmentations['rotate_right'](image)
            applied_augmentations.append('rotate_right')
            rotated_image.save(f'{save_dir}/{image_id}_rotate_right.png')

    # For factor 2 or more, apply both rotations (left and right) if not done already
    if augmentation_factor >= 2:
        if 'rotate_left' not in applied_augmentations:
            rotated_left_image = augmentations['rotate_left'](image)
            rotated_left_image.save(f'{save_dir}/{image_id}_rotate_left.png')
            applied_augmentations.append('rotate_left')

        if 'rotate_right' not in applied_augmentations:
            rotated_right_image = augmentations['rotate_right'](image)
            rotated_right_image.save(f'{save_dir}/{image_id}_rotate_right.png')
            applied_augmentations.append('rotate_right')

    # Apply flip augmentation for factors 3 and above if the image doesn't contain Cardiomegaly or Pneumothorax
    if augmentation_factor >= 3 and 'Cardiomegaly' not in disease_labels and 'Pneumothorax' not in disease_labels:
        flipped_image = augmentations['flip'](image)
        flipped_image.save(f'{save_dir}/{image_id}_flip.png')
        applied_augmentations.append('flip')

    # For factor 4 or more, apply an additional random rotation after flipping (if allowed)
    if augmentation_factor >= 4 and 'Cardiomegaly' not in disease_labels and 'Pneumothorax' not in disease_labels:
        extra_rotation = augmentations['rotate_left'](flipped_image) if np.random.rand() > 0.5 else augmentations['rotate_right'](flipped_image)
        extra_rotation.save(f'{save_dir}/{image_id}_extra_rotate.png')
        applied_augmentations.append('extra_rotation_after_flip')

    # For factor 5, apply both rotations again after flipping (if allowed)
    if augmentation_factor == 5 and 'Cardiomegaly' not in disease_labels and 'Pneumothorax' not in disease_labels:
        rotated_left_after_flip = augmentations['rotate_left'](flipped_image)
        rotated_left_after_flip.save(f'{save_dir}/{image_id}_rotate_left_after_flip.png')
        applied_augmentations.append('rotate_left_after_flip')

        rotated_right_after_flip = augmentations['rotate_right'](flipped_image)
        rotated_right_after_flip.save(f'{save_dir}/{image_id}_rotate_right_after_flip.png')
        applied_augmentations.append('rotate_right_after_flip')

    print(f'Applied augmentations for image {image_id}: {", ".join(applied_augmentations)}')

