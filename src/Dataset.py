import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image


# Custom Dataset Class
class ChestXRayDataset(Dataset):
    def __init__(self, csv_file, transform=True):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # Define diseases that should receive augmentation
        self.augmented_diseases = {}

        # Define augmentation transformation (only for training)
        self.augmentation_transform = T.Compose(
            [
                T.RandomRotation(degrees=10),
                T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                T.GaussianBlur(kernel_size=3, sigma=(0.05, 0.15)),
            ]
        )

        # Standard preprocessing for all images
        self.base_transform = T.Compose(
            [
                T.Resize((256, 256)),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485], std=[0.229]
                ),  # Normalize using standard ImageNet values
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["IMGPATH"]
        image = Image.open(img_path).convert("L")  # Convert to grayscale
        label = torch.tensor(self.data.iloc[idx]["HOTLABEL"], dtype=torch.long)

        # Apply augmentation only to selected diseases in training mode
        if (
            self.transform
            and self.data.iloc[idx]["DISEASELABEL"] in self.augmented_diseases
        ):
            image = self.augmentation_transform(image)

        # Apply base transformation to all images
        image = self.base_transform(image)

        return image, label
