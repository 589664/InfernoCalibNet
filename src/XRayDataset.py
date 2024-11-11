import torch
import numpy as np
from .utils import load_image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class XRayDataset(Dataset):
    def __init__(self, dataframe, image_dir, img_size, mean, std):
        """
        Args:
        - dataframe (pd.DataFrame): DataFrame containing ImageID, Labels, and other metadata.
        - image_dir (str): Directory with all the images.
        - img_size (int): The size to which all images will be resized (img_size x img_size).
        - mean (float): The mean pixel value for normalization.
        - std (float): The standard deviation of pixel values for normalization.
        """
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.img_size = img_size
        self.mean = mean
        self.std = std

        # Transformation to be applied (resize, convert to tensor, normalize)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[self.mean], std=[self.std])  # Use computed mean/std
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Get image path and labels
        image_id = self.dataframe.iloc[idx]['ImageID']
        img_path = f"{self.image_dir}/{image_id}.png"  # Images in PNG format

        # Load image using the helper method and convert to grayscale
        image = load_image(img_path, self.img_size)

        # Convert to numpy array and apply the transformations
        if self.transform:
            image = self.transform(image)

        # Get the multi-hot encoded labels
        labels = torch.tensor(self.dataframe.iloc[idx]['MultiHotLabels'], dtype=torch.float32)

        return image, labels

