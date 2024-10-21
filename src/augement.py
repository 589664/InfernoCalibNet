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



